import ezdxf
import pytest
from pathlib import Path
from core.paths import CAD_DIR

def test_dxf():
    dxf_path = CAD_DIR / "【CAD用実寸】コックピットフェアリング_CST_kle1.2.dxf"
    if not dxf_path.is_file():
        pytest.skip("Optional historical CAD artifact is not included in the source package")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    entities = list(msp)
    assert len(entities) > 0, "No entities found"
    units = doc.header.get("$INSUNITS")
    print(f"PASS: DXF loaded successfully. INSUNITS={units} (4=mm), Entities count={len(entities)}")
    layer_names = [l.dxf.name for l in doc.layers]
    print(f"Layers: {layer_names}")

if __name__ == "__main__":
    test_dxf()
