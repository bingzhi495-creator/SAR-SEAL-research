import ast
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import accept_float_e24 as validator


class ClosedScopeTest(unittest.TestCase):
    def seam(self, directory, scope, *, digest=None):
        source=Path(validator.__file__)
        tree=ast.parse(source.read_text(encoding="utf-8"))
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="accept_arm")
        index=next(i for i,node in enumerate(function.body)
                   if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="scope")
        code=compile(ast.Module(body=function.body[index:index+3],type_ignores=[]),str(source),"exec")
        actual=directory/"ACTUAL"
        actual.mkdir(exist_ok=True)
        lock=directory/"scope.json"
        raw=json.dumps(scope).encode()
        lock.write_bytes(raw)
        replay=[]

        def pinned(path, expected):
            data=path.read_bytes()
            if sha256(data).hexdigest()!=expected:
                raise ValueError("scope digest")
            return data

        env=dict(entry=SimpleNamespace(document=json.loads,verify_replay_receipt=replay.append),
                 m=SimpleNamespace(raw_pinned=pinned),scope_path=lock,actual=actual,Path=Path,
                 launch={"stock_scope":{"sha256":digest or sha256(raw).hexdigest()}},
                 ctx=SimpleNamespace(cfg={"coco":{"sha256":"f"*64}}),args=SimpleNamespace(arm="native"),
                 require=validator.require,same_fields=validator.same_fields)
        exec(code,env)
        self.assertEqual(replay,[scope])

    def test_requires_pinned_scope_even_when_closed_output_exists(self):
        with TemporaryDirectory() as tmp:
            directory=Path(tmp)
            scope=dict(operation="short",mode="native",coco={"sha256":"f"*64},output=str(directory/"ACTUAL"))
            self.seam(directory,scope)
            with self.assertRaisesRegex(ValueError,"scope digest"):
                self.seam(directory,scope,digest="a"*64)

    def test_rejects_wrong_closed_mode_output_or_coco(self):
        with TemporaryDirectory() as tmp:
            directory=Path(tmp)
            scope=dict(operation="short",mode="native",coco={"sha256":"f"*64},output=str(directory/"ACTUAL"))
            for different in (dict(mode="center"),dict(output=str(directory/"elsewhere")),dict(coco={})):
                with self.subTest(different=different):
                    with self.assertRaisesRegex(ValueError,"immutable closed stock scope"):
                        self.seam(directory,dict(scope,**different))


if __name__=="__main__":
    unittest.main()
