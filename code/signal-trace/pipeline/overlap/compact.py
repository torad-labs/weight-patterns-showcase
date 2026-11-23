"""Store member-level derived statistics losslessly as arrays, keep readable indexes."""
import argparse
from pathlib import Path
import numpy as np
from pipeline.overlap.analyze import read,write
from pipeline.research_guard.verify_round2 import verify,ROOT


def pack(path,fields):
    data=read(path); arrays={}
    for i,row in enumerate(data['rows']):
        for name in fields:
            if name in row and not isinstance(row[name],dict):
                key=f"position_{i}_{name}"
                arrays[key]=np.asarray(row[name],dtype='f8')
    if not arrays:return
    target=path.with_name('member-statistics.npz')
    np.savez_compressed(target,**arrays)
    with np.load(target) as restored:
        for key,value in arrays.items():
            if not np.array_equal(value,restored[key],equal_nan=True):raise ValueError('Compaction changed a statistic')
    for i,row in enumerate(data['rows']):
        for name in fields:
            if name in row and not isinstance(row[name],dict):row[name]=dict(array_file=target.name,array_key=f"position_{i}_{name}")
    data['member_statistics_storage']='Original numeric arrays moved losslessly to the named NPZ after read-back equality; the readable index and member order remain.'
    write(path,data)


def main(output):
    problems=verify(ROOT)
    if problems:raise ValueError(problems)
    output=Path(output)
    for path in sorted((output/'groups').glob('*/index.json')):pack(path,['leave_one_out_cosine'])
    fields=['failure_margin','success_margin','failure_loo_difference_cosine','success_loo_difference_cosine']
    for directory in ['contrasts','common-coordinates']:
        for path in sorted((output/directory).glob('gpu*/index.json')):pack(path,fields)
    # Correct the original descriptive label without changing numeric values.
    for path in sorted((output/'contrasts').glob('gpu*/index.json')):
        data=read(path)
        for summary in data['summaries']:
            if 'recurring_squared_by_merge' in summary:
                summary['whole_vector_supported_squared_by_merge']=summary.pop('recurring_squared_by_merge')
        data['interpretation_correction']='Whole-vector margins cannot exclude singleton components. Their numeric results are preserved as descriptive support, not a recurring target. Common coordinate portions are in common-coordinates/.'
        write(path,data)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();main(a.output)
