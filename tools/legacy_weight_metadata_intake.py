# -*- coding: utf-8 -*-
"""Read-only checkpoint container/pickle-opcode metadata. Never unpickle tensors."""
import pathlib,zipfile,pickletools,hashlib,sys,struct,re
MAX_PICKLE_BYTES=8*1024*1024
MAX_MEMBERS=50000
MAX_DIRECTORY_BYTES=16*1024*1024
KEY_LIKE=re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*\.(?:weight|bias|running_mean|running_var|num_batches_tracked)$')

def _norm(p):return str(pathlib.Path(p)).replace('\\','/').lower()

def _directory_budget(path,max_directory_bytes):
 with path.open('rb') as f:
  f.seek(0,2);size=f.tell();n=min(size,65557);f.seek(size-n);tail=f.read(n)
 pos=tail.rfind(b'PK\x05\x06')
 if pos<0 or pos+22>len(tail):raise ValueError('No complete ZIP end directory record')
 signature,disk,cd_disk,on_disk,total,cd_size,cd_offset,comment=struct.unpack('<4s4H2LH',tail[pos:pos+22])
 if pos+22+comment!=len(tail) or disk!=0 or cd_disk!=0 or on_disk!=total:raise ValueError('Unsupported multi-disk or invalid ZIP end record')
 if total==65535 or cd_size==0xffffffff or cd_offset==0xffffffff:raise ValueError('ZIP64 sentinel not parsed by this bounded intake')
 if total>MAX_MEMBERS or cd_size>max_directory_bytes:raise ValueError('Central directory budget exceeded')
 if cd_offset+cd_size>size-n+pos:raise ValueError('Central directory range invalid')
 return {'entries':total,'bytes':cd_size,'offset':cd_offset,'max_bytes':max_directory_bytes}

def inspect_archive(path,protected,max_directory_bytes=MAX_DIRECTORY_BYTES):
 p=pathlib.Path(path)
 for candidate in [p,p.resolve(strict=True)]:
  if _norm(candidate) in protected:raise ValueError('Protected original/resolved path')
 before=p.stat()
 with p.open('rb') as f:magic=f.read(16)
 result={'path':str(p),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'first16_hex':magic.hex(),'tensor_values_read':False,'checkpoint_unpickled':False,'model_loaded':False,'whole_weight_hashed':False,'opened_archive_members':[]}
 if not magic.startswith(b'PK\x03\x04'):
  result['status']='NON_ZIP_HEADER_ONLY_NOT_INTERPRETED'
 else:
  directory=_directory_budget(p,max_directory_bytes)
  with zipfile.ZipFile(str(p),'r') as z:
   entries=z.infolist()
   if len(entries)!=directory['entries']:raise ValueError('Directory entry count mismatch')
   candidates=[e for e in entries if not e.is_dir() and (e.filename=='data.pkl' or e.filename.endswith('/data.pkl'))]
   if len(candidates)!=1:raise ValueError('Exactly one pickle metadata member required')
   e=candidates[0]
   if e.file_size>MAX_PICKLE_BYTES or e.compress_size>MAX_PICKLE_BYTES or e.flag_bits&1:raise ValueError('Oversize/encrypted metadata')
   with z.open(e,'r') as f:meta=f.read(MAX_PICKLE_BYTES+1)
   if len(meta)!=e.file_size or len(meta)>MAX_PICKLE_BYTES:raise ValueError('Incomplete/oversize pickle metadata')
   strings=[];globals_=[];ops=0;seen=set();total_chars=0;unknown_globals=0
   for op,arg,pos in pickletools.genops(meta):
    ops+=1
    if ops>500000:raise ValueError('Pickle opcode budget exceeded')
    if op.name in ('UNICODE','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','STRING','BINSTRING','SHORT_BINSTRING') and isinstance(arg,str):
     if len(arg)>65536:raise ValueError('Single pickle string budget exceeded')
     if KEY_LIKE.fullmatch(arg) and arg not in seen:
      seen.add(arg);strings.append(arg);total_chars+=len(arg)
      if len(strings)>10000 or total_chars>1024*1024:raise ValueError('Key-like report output budget exceeded')
    if op.name=='GLOBAL':
     if re.fullmatch(r'(?:torch(?:\.[A-Za-z0-9_]+)*|collections|builtins|__builtin__) [A-Za-z_][A-Za-z0-9_]*',arg):globals_.append(arg)
     else:unknown_globals+=1
   result.update({'status':'ZIP_PICKLE_OPCODE_METADATA_ONLY','central_directory':directory,'archive_member_count':len(entries),'opened_archive_members':[e.filename],'pickle_metadata_bytes':len(meta),'pickle_metadata_sha256':hashlib.sha256(meta).hexdigest(),'pickle_opcode_count':ops,'metadata_strings':strings,'key_like_strings':strings,'key_like_strings_may_be_values_not_dictionary_keys':True,'arbitrary_string_values_not_reported':True,'pickle_globals':list(dict.fromkeys(globals_)),'unreported_unknown_GLOBAL_opcodes':unknown_globals,'pickle_globals_are_data_not_imported':True,'state_dict_keys_not_structurally_parsed':True,'archive_tensor_members_opened':False})
 after=p.stat()
 if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('Checkpoint changed during intake')
 if any(x in sys.modules for x in ['torch','numpy']):raise RuntimeError('Forbidden scientific import')
 return result
