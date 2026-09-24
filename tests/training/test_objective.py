import math
import pytest

torch = pytest.importorskip("torch")
from jevembed.config import ScoringConfig
from jevembed.scoring import softmax, sigmoid
from jevembed.training.objective import JevCollator, JevLoss, task_loss


@pytest.mark.parametrize("path,mode,target", [("choice",0,[1,0]),("score",0,[.2,.8]),
                                            ("score",1,[.4]),("noul_with_criteria",2,[1]),("noul_without_criteria",2,[0])])
def test_loss_matches_inference_math_and_backprop(path,mode,target):
    vectors=torch.tensor([[1.,0.],[.6,.8],[0.,1.]],requires_grad=True)
    if path=="noul_without_criteria":vectors=vectors[:2]
    scoring=ScoringConfig()
    loss=task_loss(vectors,path,mode,torch.tensor(target),scoring)
    probs=softmax([.6,0],getattr(scoring,f"{path}_temperature",scoring.choice_temperature))
    if path.startswith("noul"):
        params=getattr(scoring,path)
        p=sigmoid(params.slope*.6+params.intercept)
        expected=-target[0]*math.log(p)-(1-target[0])*math.log1p(-p)
    elif mode==1:expected=(probs[1]-target[0])**2
    else:expected=-sum(p*math.log(q) for p,q in zip(target,probs))
    assert loss.item()==pytest.approx(expected,abs=1e-6)
    gradient=torch.autograd.grad(loss,vectors)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0


def test_groups_do_not_create_cross_request_negatives():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v=torch.nn.Parameter(torch.tensor([[1.,0.],[.6,.8],[0.,1.],[.8,.6],[1.,0.]]))
        def forward(self, features):return {"sentence_embedding": self.v[features["input_ids"].flatten()]}
        def tokenize(self,texts):return {"input_ids":torch.tensor([[int(t)] for t in texts])}
    model=Model()
    rows=[{"texts":["0","1","2"],"path":0,"target_mode":0,"target":[1.,0.]},
          {"texts":["3","4"],"path":3,"target_mode":2,"target":[.3]}]
    batch=JevCollator(model)(rows)
    loss=JevLoss(model,ScoringConfig())([{"input_ids":batch["sentence_input_ids"]}],batch["label"])
    expected=(task_loss(model.v[:3],"choice",0,torch.tensor([1.,0.]),ScoringConfig())+
              task_loss(model.v[3:],"noul_without_criteria",2,torch.tensor([.3]),ScoringConfig()))/2
    assert loss.item()==pytest.approx(expected.item())
    loss.backward()
    assert model.v.grad is not None
