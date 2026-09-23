from copy import deepcopy
import gzip
import json

import pytest
from jevembed import ModelConfig, ValidationError
from jevembed.training.open_jev import convert_record, prepare, DEFAULT_REVISION, DEFAULT_SUBSET
from jevembed.training.data import load_examples


def row(kind="choice", split="train"):
    return {"id":split+kind,"group_id":split,"split":split,"state":{"visible":"text"},
            "question":"Original question", "kind":kind,"options":["no","yes"] if kind=="noul" else ["label: original text","other"],
            "target":[.25,.75],"metadata":{"privileged":"never send this", "score_values":[0,1]}}


@pytest.mark.parametrize("kind", ["choice","score","noul"])
def test_mapping_preserves_soft_labels_without_metadata(kind, tmp_path):
    source=row(kind);record=convert_record(source,"train")
    p=tmp_path/'rows.jsonl';p.write_text(json.dumps(record)+'\n')
    e=load_examples(p,ModelConfig())[0]
    assert all("privileged" not in t and "never send" not in t for t in e.texts)
    if kind=='noul':
        assert e.target==[.75] and e.plan.path=='noul_similarity'
    else:
        assert e.target==[.25,.75] and e.texts[1:]==source['options']
    changed=deepcopy(source);changed['metadata']['privileged']='something else'
    assert convert_record(changed,'train')==record


def test_noul_target_uses_yes_label_not_position():
    source=row('noul');source['options']=['yes','no']
    assert convert_record(source,'train')['answers']['decision']=={'noul':.25}


def test_reject_multilabel_and_non_index_scores():
    source=row();source['target']=[1.,1.]
    with pytest.raises(ValidationError,match='multilabel'):convert_record(source,'train')
    source=row('score');source['metadata']['score_values']=[-1,1]
    with pytest.raises(ValidationError,match='Non-index'):convert_record(source,'train')


def test_prepare_removes_exact_overlap_and_checks_groups(tmp_path):
    train=row();validation=row(split='validation')
    other=deepcopy(train);other['id']='unique';other['state']='new state'
    paths={s:tmp_path/(s+'.gz') for s in ['train','validation']}
    for s,rows in [('train',[train,other]),('validation',[validation])]:
        with gzip.open(paths[s],'wt') as f:
            f.write(''.join(json.dumps(r)+'\n' for r in rows))
    m=prepare(paths,tmp_path/'prepared',DEFAULT_REVISION,DEFAULT_SUBSET,deduplicate=True)
    assert m['splits']['train']['cross_split_inputs_removed']==1
    assert m['splits']['train']['retained_questions']==1
