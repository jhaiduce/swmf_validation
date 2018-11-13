import unittest

class test_build_imfinput(unittest.TestCase):

    def test_write_imfinput(self):
        import subprocess

        subprocess.call(['python','-m','swmf_validation.build_imfinput','2018-01-01','2018-01-02','2018-01-01.dat'])

        from difflib import Differ

        referenceLines=open('data/2018-01-01.dat').readlines()
        testLines=open('2018-01-01.dat').readlines()
        result=list(Differ().compare(referenceLines,testLines))
        import re
        self.assertEqual(result[0][:2],'- ')
        self.assertEqual(result[2][:2],'+ ')
        for line in result[0],result[2]:
            self.assertTrue(re.match('File created on \d{4}-\d{2}-\d{2}T\d\d:\d\d:\d\d(\.\d+)?',line[2:]))

        for line in result[4:]:
            self.assertEqual(line[:2],'  ')

if __name__=='__main__':
    unittest.main()
