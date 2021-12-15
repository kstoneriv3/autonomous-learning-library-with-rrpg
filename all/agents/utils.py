import types
import torch

def _reinforce_policy(self, loss, return_grad=False):
    loss = self._loss_scaling * loss
    self._writer.add_loss(self._name, loss.detach())
    loss.backward()

    if return_grad:
        grads = [p.grad.view(-1).detach().clone() for p in self.model.parameters()]
        self.step()
        return self, grads
    else:
        self.step()
        return self

def _reinforce_features(self, return_grad=False):
    graphs, grads = self._dequeue()
    if graphs.requires_grad:
        graphs.backward(grads)

    if return_grad:
        grads = [p.grad.view(-1).detach().clone() for p in self.model.parameters()]
        self.step()
        return self, grads
    else:
        self.step()
        return self

def make_grads_observable(agent):
    # update policy and feature network so that 
    # policy.reinforce() and feature.reinforce()
    # returns gradient information when requested in the argument.
    agent.policy.reinforce = types.MethodType(_reinforce_policy, agent.policy)
    agent.features.reinforce = types.MethodType(_reinforce_features, agent.features)

def flatten_grads(sequence):
    return torch.cat([t.reshape(-1) for t in sequence]) if len(sequence) > 0 else torch.tensor([])

