import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def rebind(snapshot: Path):
    tree = ET.parse(snapshot)
    for kind in ("midi", "audio"):
        paths = sorted(snapshot.parent.glob(f"interchange/*/{kind}files"))
        option = tree.find(f'Config/Option[@name="{kind}-search-path"]')
        if option is not None:
            option.set("value", ":".join(str(path.resolve()) for path in paths))
    tree.write(snapshot, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    rebind(parser.parse_args().snapshot)
