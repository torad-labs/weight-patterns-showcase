"""A group contrast retains the vector at each connected step, not just its norm."""
import json
import unittest
import numpy as np
from tests import test_compare_occurrences as fixtures
from pipeline.native_records.group_observation import GroupObservation
from pipeline.observation.compare_a_groups import compare_groups


class GroupContrastTests(unittest.TestCase):
    setUp = fixtures.ConnectedComparisonTests.setUp
    tearDown = fixtures.ConnectedComparisonTests.tearDown
    fixture = fixtures.ConnectedComparisonTests.fixture
    mapping = fixtures.ConnectedComparisonTests.mapping

    def make_group(self, label, values):
        paths=[self.fixture(label+str(i), [x, [2*v for v in x], [3*v for v in x], [5*v for v in x]])
               for i,x in enumerate(values)]
        group=GroupObservation.create(self.root/label, paths[0]/'signal-events.jsonl', dict(gpu=0,role=label))
        for i,p in enumerate(paths):
            group.add(label+str(i),p/'signal-events.jsonl',p/'signal-data.bin',self.mapping())
        group.finish()
        return group

    def test_keeps_changes_at_all_steps_and_actual_merge_connections(self):
        left=self.make_group('left',[[1,2],[2,1]])
        right=self.make_group('right',[[2,3],[3,2]])
        out=self.root/'contrast'
        report=compare_groups(left,right,self.mapping(),out)
        rows=[json.loads(line) for line in (out/'locations.jsonl').read_text().splitlines()]
        data=np.fromfile(out/'contrasts.f32',dtype='<f4')
        np.testing.assert_array_equal(data[rows[0]['offset']:rows[0]['offset']+2],[2,2])
        np.testing.assert_array_equal(data[rows[1]['offset']:rows[1]['offset']+2],[3,3])
        np.testing.assert_array_equal(data[rows[2]['offset']:rows[2]['offset']+2],[5,5])
        add=rows[2]['addition_change']
        np.testing.assert_array_equal(data[add['offset']:add['offset']+2],[3,3])
        self.assertEqual(rows[2]['left_count'],2)
        self.assertEqual(rows[2]['right_count'],2)
        self.assertEqual(report['compared_locations'],3)
        self.assertEqual(rows[2]['connection'],{'left_execution':1,'right_execution':1,'local_operation':2})

    def test_different_coordinate_shapes_remain_unmatched(self):
        left=self.make_group('left',[[1,2],[2,1]])
        right=self.make_group('right',[[2,3,4],[3,2,4]])
        out=self.root/'contrast'
        report=compare_groups(left,right,self.mapping(),out)
        self.assertEqual(report['compared_locations'],0)
        self.assertEqual(report['unmatched_locations'],3)
        self.assertEqual((out/'contrasts.f32').stat().st_size,0)


if __name__=='__main__':
    unittest.main()
