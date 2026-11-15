"""Contract and integration tests for proposal-faithful CF-CAH."""
import types
import unittest

import chess
import torch
import torch.nn.functional as F

from maia3.action_head import CFActPolicyHead
from maia3.action_head_cah import CFCahPolicyHead
from maia3.action_to_board import ActionToBoardCrossAttention
from maia3.actions import collate_legal_actions, legal_moves_to_actions
from maia3.model_registry import apply_model_config, resolve_model_spec
from maia3.models import MAIA3Model
from maia3.models_cf_cah import MAIA3CahModel


class TestAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.block = ActionToBoardCrossAttention(32, 4).eval()
        self.board = torch.randn(3, 64, 32)
        self.m0 = torch.randn(3, 5, 32)
        self.mask = torch.tensor([[1,1,1,1,0],[1,0,0,0,0],[0,0,0,0,0]], dtype=torch.bool)

    def test_shapes_axis_padding_and_terminal(self):
        c, a = self.block(self.m0, self.board, self.mask, True)
        self.assertEqual(c.shape, (3, 5, 32))
        self.assertEqual(a.shape, (3, 4, 5, 64))
        self.assertTrue(torch.allclose(a.sum(-1)[0,:,:4], torch.ones(4,4), atol=1e-6))
        self.assertEqual(torch.count_nonzero(a[2]).item(), 0)
        self.assertEqual(torch.count_nonzero(c[~self.mask]).item(), 0)
        self.assertTrue(torch.isfinite(c).all())

    def test_no_action_to_action_dependency(self):
        other = self.m0.clone()
        other[0,0] += 10
        c1, _ = self.block(self.m0, self.board, self.mask)
        c2, _ = self.block(other, self.board, self.mask)
        self.assertFalse(torch.equal(c1[0,0], c2[0,0]))
        self.assertTrue(torch.equal(c1[0,1:], c2[0,1:]))

    def test_parameter_count_and_no_kv_norm(self):
        self.assertEqual(sum(p.numel() for p in self.block.parameters()), 8*32*32 + 7*32)
        self.assertFalse(any("ln_kv" in name for name in self.block.state_dict()))
        with self.assertRaises(ValueError):
            ActionToBoardCrossAttention(30, 8)


class TestPolicy(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.board = torch.randn(3, 64, 32)
        self.src = torch.tensor([[2,3,4],[7,0,0],[0,0,0]])
        self.dst = torch.tensor([[8,9,10],[15,0,0],[0,0,0]])
        self.promo = torch.zeros_like(self.src)
        self.mask = torch.tensor([[1,1,1],[1,0,0],[0,0,0]],dtype=torch.bool)

    def test_policy_probability_and_attention(self):
        h = CFCahPolicyHead(32, num_heads=4).eval()
        r = h(self.board,self.src,self.dst,self.promo,self.mask,return_details=True,return_attention=True)
        self.assertEqual(r.attention_weights.shape,(3,4,3,64))
        self.assertTrue(torch.isneginf(r.logits_legal[~self.mask]).all())
        self.assertTrue(torch.all(r.policy_probs[~self.mask]==0))
        self.assertTrue(torch.allclose(r.policy_probs.sum(-1),torch.tensor([1.,1.,0.])))
        self.assertFalse(torch.isnan(r.policy_probs).any())
        self.assertEqual(len(h(self.board,self.src,self.dst,self.promo,self.mask)),3)

    def test_zero_attention_identity_matches_act(self):
        act = CFActPolicyHead(32).eval()
        cah = CFCahPolicyHead(32,num_heads=4).eval()
        cah.load_state_dict(act.state_dict(),strict=False)
        with torch.no_grad():
            cah.cross_attention.out_proj.weight.zero_()
            cah.cross_attention.ffn[3].weight.zero_()
            cah.cross_attention.ffn[3].bias.zero_()
        for x,y in zip(act(self.board,self.src,self.dst,self.promo,self.mask),
                       cah(self.board,self.src,self.dst,self.promo,self.mask)):
            self.assertTrue(torch.equal(x,y))

    def test_gradient_flow(self):
        h=CFCahPolicyHead(32,num_heads=4).train()
        logits,_,_=h(self.board,self.src,self.dst,self.promo,self.mask)
        F.cross_entropy(logits[:2],torch.tensor([0,0])).backward()
        for name in ("q_proj","k_proj","v_proj","out_proj"):
            g=getattr(h.cross_attention,name).weight.grad
            self.assertIsNotNone(g,name)
            self.assertTrue(torch.isfinite(g).all(),name)
            self.assertGreater(g.abs().sum().item(),0,name)
        self.assertGreater(h.cross_attention.ffn[0].weight.grad.abs().sum().item(),0)


class TestFullModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = types.SimpleNamespace()
        apply_model_config(cls.cfg,resolve_model_spec("maia3-5m"))

    def test_backbone_wdl_ponder_parity(self):
        torch.manual_seed(43)
        base=MAIA3Model(self.cfg).eval()
        cah=MAIA3CahModel(self.cfg).eval()
        src=base.state_dict();dst=cah.state_dict()
        for key in src.keys() & dst.keys():
            dst[key].copy_(src[key])
        cah.load_state_dict(dst)
        x=torch.randn(2,64,12*self.cfg.history+1)
        elo=torch.tensor([1200,1600])
        with torch.no_grad():
            _,v,p=base(x,elo,elo)
            _,v2,p2=cah.encode_board(x,elo,elo)
        self.assertTrue(torch.allclose(v,v2,atol=1e-5))
        self.assertTrue(torch.allclose(p,p2,atol=1e-5))

    def test_terminal_and_4352(self):
        model=MAIA3CahModel(self.cfg).eval()
        boards=[
            chess.Board(),
            chess.Board("r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"),
            chess.Board("k7/8/1Q6/8/8/8/8/7K b - - 0 1"),
        ]
        self.assertTrue(boards[1].is_checkmate())
        self.assertTrue(boards[2].is_stalemate())
        a=collate_legal_actions([legal_moves_to_actions(b) for b in boards])
        x=torch.zeros(3,64,12*self.cfg.history+1)
        elo=torch.tensor([1500,1500,1500])
        with torch.no_grad():
            r=model.forward_with_details(x,elo,elo,a["src_sq"],a["dst_sq"],
                a["promo_type"],a["action_mask"],return_attention=True)
            logits,v,p=model(x,elo,elo,boards,True)
        self.assertTrue(torch.all(r.policy_probs[1:]==0))
        self.assertAlmostEqual(r.policy_probs[0].sum().item(),1,places=6)
        self.assertEqual(logits.shape,(3,4352))
        self.assertTrue(torch.isneginf(logits[1:]).all())
        self.assertEqual(v.shape,(3,3))
        self.assertEqual(p.shape,(3,))

    def test_inherited_forward_contract(self):
        model=MAIA3CahModel(self.cfg).eval()
        a=collate_legal_actions([legal_moves_to_actions(chess.Board())])
        x=torch.zeros(1,64,12*self.cfg.history+1)
        elo=torch.tensor([1500])
        with torch.no_grad():
            outputs=model.forward_with_actions(x,elo,elo,a["src_sq"],a["dst_sq"],
                a["promo_type"],a["action_mask"])
        self.assertEqual(len(outputs),5)
        self.assertEqual(outputs[2].shape,(1,3))
        self.assertEqual(outputs[3].shape,(1,))


if __name__ == "__main__":
    unittest.main()
