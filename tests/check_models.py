"""快速验证指定模型的输出(开发辅助,不属于正式测试)。用法: python tests/check_models.py model_id ..."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import open_spss.models  # noqa: F401
from open_spss.core.registry import REGISTRY
from open_spss.core.datasets import STORE
from test_smoke_all_models import default_slots, ds_id

for mid in sys.argv[1:]:
    spec = REGISTRY.get(mid)
    ds = STORE.get(ds_id(spec.demo_note or "survey"))
    slots = default_slots(spec, ds)
    opts = {o.key: o.default for o in spec.options}
    slots, opts = REGISTRY.validate(mid, slots, opts)
    r = spec.runner(ds, slots, opts).to_dict()
    interp = (r["interpretation"] or [""])[0][:70]
    print(f"OK {mid}: tables={len(r['tables'])} charts={len(r['charts'])} | {interp}")
