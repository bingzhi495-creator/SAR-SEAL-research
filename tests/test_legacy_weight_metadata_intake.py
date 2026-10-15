# -*- coding: utf-8 -*-
import importlib.util, pathlib, pickle, tempfile, unittest, zipfile
SOURCE=pathlib.Path(__file__).resolve().parents[1]/'tools'/'legacy_weight_metadata_intake.py'
spec=importlib.util.spec_from_file_location('legacy_metadata_intake',str(SOURCE))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class MetadataIntakeBehavior(unittest.TestCase):
 def test_pickle_metadata_is_described_without_executing_reduce(self):
  with tempfile.TemporaryDirectory() as td:
   td=pathlib.Path(td);marker=td/'MUST_NOT_EXIST.txt'
   class ExecutablePickle:
    def __reduce__(self):return (eval,("__import__('pathlib').Path(%r).write_text('executed')"%str(marker),))
   p=td/'sample.pth'
   with zipfile.ZipFile(str(p),'w') as z:
    z.writestr('archive/data.pkl',pickle.dumps({'conv0_0.conv.0.weight':ExecutablePickle()},protocol=2))
    z.writestr('archive/data/0',b'not-a-tensor-this-must-not-be-opened')
   result=m.inspect_archive(p,set())
   self.assertFalse(marker.exists())
   self.assertEqual(result['status'],'ZIP_PICKLE_OPCODE_METADATA_ONLY')
   self.assertEqual(result['opened_archive_members'],['archive/data.pkl'])
   self.assertIn('conv0_0.conv.0.weight',result['metadata_strings'])
   self.assertIn('__builtin__ eval',result['pickle_globals'])
   self.assertFalse(result['tensor_values_read'])

 def test_container_directory_budget_is_enforced_before_pickle_read(self):
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td)/'large-directory.pth'
   with zipfile.ZipFile(str(p),'w') as z:
    z.writestr('archive/data.pkl',pickle.dumps({'conv.weight':1},protocol=2))
    z.writestr('archive/data/0',b'not-a-tensor')
   with self.assertRaisesRegex(ValueError,'Central directory budget'):
    m.inspect_archive(p,set(),max_directory_bytes=64)

if __name__=='__main__':unittest.main()
