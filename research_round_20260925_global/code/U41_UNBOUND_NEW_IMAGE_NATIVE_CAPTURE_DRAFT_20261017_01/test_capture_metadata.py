"""Three stdlib interface tests, not numerical or image qualification."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


def subject():
    path=Path(__file__).with_name('native_capture.py')
    spec=importlib.util.spec_from_file_location('_u41_capture_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class MetadataTests(unittest.TestCase):
    def test_image_only_loader_does_not_request_annotation_or_parser(self):
        m=subject(); seen=[]
        class Poison:
            def __getattribute__(self,key):raise AssertionError('GT poison accessed')
        class Sample:
            ori_shape=(283,503);img_shape=(450,800);scale_factor=(800/503,450/283)
            def __init__(self):self.targets={'gt_instances':Poison(),'ignored_instances':Poison()}
            def __contains__(self,key):return key in self.targets
            def __delattr__(self,key):self.targets.pop(key)
            def set_metainfo(self,value):self.record=dict(value)
        class Pipeline:
            def __init__(self,steps):seen.extend(steps)
            def __call__(self,info):
                self_info=dict(info);seen.append(self_info)
                return {'img_id':info['img_id'],'ori_shape':(283,503),'img_shape':(450,800),'img':'pixels'}
        class Pack:
            def __init__(self,**kwargs):seen.append(kwargs)
            def __call__(self,value):return {'inputs':value['img'],'data_samples':Sample()}
        metadata=dict(image_id=871,image_ref={'path':str(Path(__file__).resolve()),'bytes':1,'sha256':'a'*64},
                      original_shape=[283,503])
        # Extra injected dependencies are never consulted by the image-only packer.
        modules=dict(Compose=Pipeline,PackDetInputs=Pack,annotations=Poison(),parser=Poison())
        packed,identity=m.pack_image(metadata,modules,lambda ref:ref['path'])
        self.assertEqual([s['type'] for s in seen[:2]],['LoadImageFromFile','Resize'])
        self.assertEqual(seen[2],{'img_id':871,'img_path':metadata['image_ref']['path']})
        self.assertEqual(identity['actual_img_shape'],[450,800]);self.assertEqual(packed['inputs'],'pixels')
        class Preprocessor:
            batch_augments=[SimpleNamespace(size=(800,800))]
            def _get_pad_shape(self,batch):return [(480,800)]
            def __call__(self,batch,training):
                self_seen=dict(batch);seen.append((self_seen,training))
                if batch['data_samples'] is not None:raise AssertionError('Target container forwarded')
                return dict(inputs=SimpleNamespace(shape=(1,3,800,800)),data_samples=None)
        modules['pseudo_collate']=lambda rows:dict(inputs=[row['inputs'] for row in rows])
        data,samples,padding=m.preprocess_images_only(SimpleNamespace(data_preprocessor=Preprocessor()),packed,identity,modules)
        self.assertEqual(seen[-1],({'inputs':['pixels'],'data_samples':None},True))
        self.assertEqual(samples[0].targets,{})
        self.assertEqual(samples[0].record,dict(batch_input_shape=(800,800),pad_shape=(800,800)))
        self.assertEqual(padding['pre_fixed_pad_shape'],[480,800])
        with self.assertRaises((ValueError,RuntimeError)):
            m.pack_image(dict(metadata,annotations=Poison()),modules,lambda ref:ref['path'])

    def test_missing_original_padding_or_photometry_is_rejected(self):
        m=subject()
        config=dict(type='DetDataPreprocessor',mean=[1.,2.,3.],std=[4.,5.,6.],bgr_to_rgb=True,
                    pad_size_divisor=32,pad_mask=True,pad_value=0,
                    batch_augments=[dict(type='BatchFixedSizePad',size=(800,800),img_pad_value=0,
                                         pad_mask=True,mask_pad_value=0,pad_seg=False)])
        m.validate_preprocessing(config)
        for key in ('batch_augments','mean','std','bgr_to_rgb','pad_value'):
            bad=dict(config);del bad[key]
            with self.subTest(key=key),self.assertRaises((ValueError,RuntimeError)):
                m.validate_preprocessing(bad)
        bad=dict(config,batch_augments=[dict(config['batch_augments'][0],size=(512,512))])
        with self.assertRaises((ValueError,RuntimeError)):m.validate_preprocessing(bad)

    def test_native_fields_use_final_mask_embed_output_not_pre_hook_h(self):
        m=subject();F=object();e=[object() for _ in range(10)];h=object()
        classes=[object() for _ in range(10)];masks=[object() for _ in range(10)]
        observed=dict(F=[F],e=e,output=[(classes,masks)],h=[h])
        fields=m.select_native_fields(observed,classes,masks)
        self.assertIs(fields['F'],F);self.assertIs(fields['e'],e[-1]);self.assertIsNot(fields['e'],h)
        self.assertIs(fields['M'],masks[-1]);self.assertIs(fields['C'],classes[-1])
        self.assertEqual(fields['natural_query_ids'],list(range(100)))
        with self.assertRaises((ValueError,RuntimeError)):
            m.select_native_fields(dict(F=[F],h=[h],output=[(classes,masks)]),classes,masks)


if __name__=='__main__':unittest.main()
