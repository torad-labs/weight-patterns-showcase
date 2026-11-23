"""The byte prefilter may only exclude files the existing text scan excludes."""
import itertools
import re
import unittest
from unittest.mock import patch

from pipeline.research_guard.forbidden_bytes import needles, without_a_reference


class StatelessForbiddenBytesTests(unittest.TestCase):
    def test_no_false_clean_result_over_valid_invalid_and_newline_inputs(self):
        references = ['blocked-name', 'old/path.py', '[retired]']
        table = needles(references)
        chunks = [b'', b'ordinary', b'blocked-name', b'old/path.py', b'[retired]',
                  b'blocked-', b'name', b'\xff', b'\xe2\x82', b'\r', b'\n', b'\r\n', 'café'.encode()]
        pattern = re.compile('|'.join(re.escape(s) for s in references))
        for left, right in itertools.product(chunks, repeat=2):
            data = left+right
            decoded = data.decode('utf-8',errors='ignore').replace('\r\n','\n').replace('\r','\n')
            if without_a_reference(data,table):
                self.assertIsNone(pattern.search(decoded),data)

    def test_invalid_bytes_that_join_a_reference_take_the_original_decoder(self):
        table=needles(['blocked-name'])
        for data in (b'blocked-\xffname', b'blocked-\xe2\x82name'):
            self.assertFalse(without_a_reference(data,table))

    def test_line_end_unicode_punctuation_and_empty_references_fall_back(self):
        for references in ([],[''],['two\nlines'],['two\rlines'],['café'],['[!]']):
            with self.subTest(references=references):
                self.assertIsNone(needles(references))
                self.assertFalse(without_a_reference(b'anything',needles(references)))

    def test_non_utf8_default_encoding_cannot_skip_the_text_reader(self):
        import pipeline.research_guard.forbidden_bytes as module
        with patch.object(module.locale,'getencoding',return_value='latin1'), patch.object(module.sys,'flags',type('Flags',(),{'utf8_mode':0})()):
            self.assertIsNone(needles(['blocked-name']))

    def test_real_reference_is_never_skipped_and_clean_utf8_can_be(self):
        table=needles(['blocked-name','blocked-other'])
        self.assertFalse(without_a_reference(b'prefix blocked-name suffix',table))
        self.assertTrue(without_a_reference('ordinary café\r\n'.encode(),table))
        self.assertTrue(without_a_reference(b'blocked but no whole reference',table))


if __name__=='__main__':
    unittest.main()
