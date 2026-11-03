"""Scientific checks for axes, unit conversion, MC energy and deterministic inputs."""
import unittest,json,glob
import numpy as np
from common import ROOT, load_config, make_model, property_table, write_json
from virtual_detectors import detector_offsets,integrate_patch

class ScientificTests(unittest.TestCase):
    def test_geometry_and_centres(self):
        c=load_config();labels,mua,axes=make_model(c)
        self.assertEqual(labels.shape,(80,80,50))
        for j,inc in enumerate(c['inclusions'],2):
            ii=np.where(labels==j)
            for k in range(3): self.assertAlmostEqual(axes[k][ii[k]].mean(),inc['centre_mm'][k],places=7)
            self.assertLess(abs(len(ii[0])-4*np.pi*5**3/3)/(4*np.pi*5**3/3),.06)
    def test_units_and_optical_property_mapping(self):
        c=load_config();p=property_table(c,'two_spheres')
        np.testing.assert_allclose(p[1:,0],[.1/10,.2/10,.05/10])
        np.testing.assert_allclose(p[1:,1]*(1-p[1:,2]),1)
        self.assertEqual(p[0,3],1)
        np.testing.assert_allclose(property_table(c,'homogeneous')[1:,0],.01)
    def test_patch_coordinates_and_out_of_field(self):
        c=load_config();a=np.ones((80,80))
        self.assertEqual(integrate_patch(a,0,0,c),9.)
        self.assertTrue(np.isnan(integrate_patch(a,40,0,c)))
        self.assertEqual(detector_offsets().shape,(56,2))
        np.testing.assert_allclose(np.linalg.norm(detector_offsets(),axis=1).reshape(7,8)[:,0],np.linspace(15,25,7))
    def test_exit_is_air_boundary_tally(self):
        a=np.load(ROOT/'data/mcx/centre_homogeneous_1000000.npz')
        self.assertEqual(a['fluence'].shape,(80,80,50))
        self.assertEqual(a['boundary_all'].shape,(82,82,52))
        np.testing.assert_allclose(a['transmission'],a['boundary_all'][1:-1,1:-1,-1])
        self.assertGreater(float(a['transmission'].sum()),0)
        self.assertFalse(np.array_equal(a['fluence'][:,:,-1],a['transmission']))
    def test_all_runs_finite_and_energy_closure(self):
        paths=list((ROOT/'data/mcx').glob('*.json'))
        summaries=[json.loads(p.read_text(encoding='utf-8')) for p in paths if not p.stem.endswith('_input')]
        self.assertGreaterEqual(len(summaries),30)
        for r in summaries:
            self.assertLess(abs(r['energy_closure']-1),.001,r['tag'])
            a=np.load(ROOT/r['data_file'])
            self.assertTrue(np.isfinite(a['transmission']).all(),r['tag'])
            self.assertGreaterEqual(float(a['transmission'].min()),0)
    def test_high_absorption_transmits_less(self):
        p=ROOT/'data/mcx'
        b=np.load(p/'centre_homogeneous_10000000.npz')['transmission'].sum(dtype=np.float64)
        h=np.load(p/'centre_high_only_10000000.npz')['transmission'].sum(dtype=np.float64)
        t=np.load(p/'centre_two_spheres_10000000.npz')['transmission'].sum(dtype=np.float64)
        self.assertLess(h,b)
        self.assertGreater(t,h)
    def test_fixed_seed_repeat_and_time_window(self):
        p=ROOT/'data/mcx'
        a=np.load(p/'centre_homogeneous_1000000.npz')['transmission']
        b=np.load(p/'repeat_fixed_seed.npz')['transmission']
        # Atomic accumulation order may differ at the final floating-point bits.
        self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(a),1e-4)
        # Extending a time cut can change later RNG histories, so raw noisy images
        # need not agree pixel by pixel. Check directly how much exits after 10 ns.
        w=json.loads((p/'window_20ns_twogates.json').read_text(encoding='utf-8'))
        gates=w['transmission_per_gate_fraction']
        self.assertEqual(len(gates),2)
        self.assertLess(gates[1]/sum(gates),.001)
    def test_changed_configuration_cannot_reuse_old_results(self):
        from mcx_forward import run_one
        c=load_config();c['background']['mua']=.012
        with self.assertRaisesRegex(RuntimeError,'different inputs'):
            run_one(c,'homogeneous',1000000,c['seed_base'],'centre_homogeneous_1000000')

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ScientificTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(ROOT/'results/validation.json',dict(tests_run=result.testsRun,failures=[str(x) for x in result.failures],errors=[str(x) for x in result.errors],passed=result.wasSuccessful()))
    if not result.wasSuccessful():raise SystemExit(1)
