# tests/test_db.py
import tempfile, os
from database import ImageDatabase
from PIL import Image
def test_db_get_filepath_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        dbp = os.path.join(d, "test.db")
        imgp = os.path.join(d, "a.jpg")
        open(imgp, "wb").close()
        db = ImageDatabase(dbp)
        db.add_image(123, imgp, 10, 10, "x")
        assert db.get_filepath(123) == imgp
        db.close()

def test_image_database_add_and_get():
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test.db")
        # sahte görseller
        img1 = os.path.join(d, "a.jpg"); Image.new("RGB",(10,10)).save(img1)
        img2 = os.path.join(d, "b.jpg"); Image.new("RGB",(10,10)).save(img2)

        db = ImageDatabase(db_path)
        db.add_image(1, img1, 10, 10, None)
        db.add_image(2, img2, 10, 10, None)

        assert db.get_filepath(1) == img1
        assert db.get_filepath(2) == img2
        assert db.get_filepath(9999) is None

        # persistency
        db.close()
        db2 = ImageDatabase(db_path)
        assert db2.get_filepath(1) == img1
        db2.close()