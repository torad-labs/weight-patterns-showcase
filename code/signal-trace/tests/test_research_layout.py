from pathlib import Path
import tempfile
import unittest

from pipeline.maintenance.research_layout import move_and_link


class ResearchLayoutTests(unittest.TestCase):
    def test_directory_move_keeps_bytes_inode_and_old_absolute_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'joint-a2';old.mkdir();p=old/'result.json';p.write_bytes(b'unchanged record')
            inode=p.stat().st_ino;new=root/'A2/full-path'
            move_and_link(old,new)
            self.assertTrue(old.is_symlink())
            self.assertEqual(p.read_bytes(),b'unchanged record')
            self.assertEqual((new/'result.json').stat().st_ino,inode)
            move_and_link(old,new)
            self.assertEqual(p.stat().st_ino,inode)

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';new=root/'new';old.write_text('one');new.write_text('two')
            with self.assertRaises(FileExistsError): move_and_link(old,new)
            self.assertEqual(old.read_text(),'one');self.assertEqual(new.read_text(),'two')
