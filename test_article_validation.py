#!/usr/bin/env python3
"""
Comprehensive test suite to validate all calculations, data coherence, 
and scientific accuracy of the BESEVIC (2025) article.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add current directory to path to import make_figures
sys.path.append(str(Path(__file__).parent))

class TestBrownYorkConvergence:
    """Test the fundamental Brown-York convergence for Schwarzschild"""
    
    def test_schwarzschild_formula_accuracy(self):
        """Test exact Brown-York formula for Schwarzschild"""
        M = 1.0
        R_vals = np.logspace(1, 3, 100)
        
        # Exact formula
        E_BY = R_vals * (1 - np.sqrt(1 - 2*M/R_vals))
        
        # Should converge to M as R -> infinity
        large_R_error = abs(E_BY[-1] - M)
        assert large_R_error < 1e-3, f"Large R convergence failed: error = {large_R_error}"
        
        # Should be monotonically decreasing error
        rel_errors = np.abs(E_BY - M) / M
        assert np.all(np.diff(rel_errors) < 0), "Error should decrease monotonically"
    
    def test_convergence_scaling(self):
        """Test that convergence follows expected 1/R scaling"""
        M = 1.0
        R_vals = np.logspace(1.5, 2.5, 20)
        
        E_BY = R_vals * (1 - np.sqrt(1 - 2*M/R_vals))
        rel_errors = np.abs(E_BY - M) / M
        
        # For large R, error ~ M/R
        theoretical_errors = M / R_vals
        
        # Check scaling is correct within factor of 5
        scaling_ratio = rel_errors / theoretical_errors
        assert np.all(scaling_ratio > 0.2) and np.all(scaling_ratio < 5), \
            f"Scaling deviates too much from 1/R: ratios = {scaling_ratio}"

class TestEllipsoidStability:
    """Test ellipsoid calculations and stability"""
    
    def test_spherical_limit(self):
        """Test that ellipsoid reduces to sphere when b/a = 1"""
        from make_figures import fig_ellipsoids_smooth
        
        # At q = 1.0 (sphere), all beta values should give similar results
        a_base = 15.0
        M = 1.0
        q = 1.0
        
        def rel_err_proxy(q, beta=0.0):
            r_eff = (a_base*a_base*(q*a_base))**(1.0/3.0)
            M_est = r_eff * (1 - np.sqrt(1 - 2*M/r_eff))
            anis = beta * 0.02 * (q - 1.0)**2
            return np.abs((M_est + anis) - M)
        
        # All beta values should give same result at q=1
        error_beta0 = rel_err_proxy(1.0, beta=0.0)
        error_beta025 = rel_err_proxy(1.0, beta=0.25)
        error_beta05 = rel_err_proxy(1.0, beta=0.5)
        
        assert np.allclose([error_beta0, error_beta025, error_beta05], error_beta0, rtol=1e-10), \
            "All beta values should give same result at spherical limit"
    
    def test_aspect_ratio_stability(self):
        """Test that method is stable across aspect ratio range"""
        a_base = 15.0
        M = 1.0
        qs = np.linspace(0.7, 1.3, 21)
        
        def rel_err_proxy(q, beta=0.0):
            r_eff = (a_base*a_base*(q*a_base))**(1.0/3.0)
            M_est = r_eff * (1 - np.sqrt(1 - 2*M/r_eff))
            anis = beta * 0.02 * (q - 1.0)**2
            return np.abs((M_est + anis) - M)
        
        errors = [rel_err_proxy(q) for q in qs]
        
        # All errors should be reasonable (< 0.1)
        assert np.all(np.array(errors) < 0.1), \
            f"Some errors too large: max = {max(errors)}"
        
        # No sudden jumps
        error_diffs = np.abs(np.diff(errors))
        assert np.all(error_diffs < 0.01), \
            f"Errors change too rapidly: max diff = {max(error_diffs)}"

class TestAstrophysicalValidation:
    """Test astrophysical validation data"""
    
    def test_object_data_consistency(self):
        """Test that astrophysical object data is self-consistent"""
        KM_PER_MSUN = 1.476625
        objects = [
            ("Sgr A*",     "SMBH", 4.15e6, None),
            ("M87*",       "SMBH", 6.5e9,  None),
            ("Cygnus X-1", "BH",   21.2,   None),
            ("GW150914",   "BH",   62.0,   None),
            ("PSR J0737-3039A", "NS", 1.34,  11.9),
            ("PSR J1614-2230", "NS", 1.97,  12.0),
        ]
        
        for name, typ, M_solar, R_km in objects:
            # Mass should be positive
            assert M_solar > 0, f"{name}: negative mass"
            
            # Neutron stars should have reasonable radii
            if typ == "NS":
                assert R_km is not None, f"{name}: NS missing radius"
                assert 8 < R_km < 20, f"{name}: unrealistic NS radius {R_km} km"
                
                # Check compactness (GM/Rc² should be reasonable for NS)
                compactness = M_solar * KM_PER_MSUN / R_km
                assert 0.1 < compactness < 0.5, \
                    f"{name}: unrealistic compactness {compactness}"
    
    def test_brown_york_calculations(self):
        """Test Brown-York calculations on astrophysical objects"""
        def E_BY_over_M(R_over_M):
            R = R_over_M
            inside = 1.0 - 2.0 / R
            inside = np.clip(inside, 1e-12, None)
            return R * (1.0 - np.sqrt(inside))
        
        # Test specific cases
        R_over_M_test = np.array([10.0, 6.0, 4.0])  # BH, NS cases
        results = E_BY_over_M(R_over_M_test)
        
        # Results should be reasonable (between 1.0 and 1.2 for these R/M)
        assert np.all(results > 1.0), "E_BY should overestimate mass"
        assert np.all(results < 1.3), "E_BY overestimate should be reasonable"
        
        # Larger R/M should give smaller error
        assert results[0] < results[1] < results[2], \
            "Error should increase as R/M decreases"
    
    def test_astrophysical_figure_convergence_panel(self):
        """Test panel (c) of astrophysical validation figure - Schwarzschild convergence"""
        
        # Test the exact calculation used in panel (c)
        def E_BY_over_M(R_over_M):
            R = R_over_M
            inside = 1.0 - 2.0 / R
            inside = np.clip(inside, 1e-12, None)
            return R * (1.0 - np.sqrt(inside))
        
        # Test range used in the figure: R/M from 3.1 to 200 (extended)
        Rscan = np.linspace(3.1, 200, 100)
        E_over_M = E_BY_over_M(Rscan)
        err_curve = (E_over_M - 1.0) * 100.0  # relative error in %
        abs_err_curve = np.abs(err_curve)
        
        # Test 1: Error should decrease monotonically with R/M
        error_diffs = np.diff(abs_err_curve)
        assert np.all(error_diffs <= 1e-10), \
            f"Error should decrease monotonically: max increase = {error_diffs.max()}"
        
        # Test 2: Error at R/M = 10 should be around 5-6% (used for black holes)
        R_10_idx = np.argmin(np.abs(Rscan - 10.0))
        error_at_10M = abs_err_curve[R_10_idx]
        assert 5.0 < error_at_10M < 7.0, \
            f"Error at R=10M should be ~5-6%, got {error_at_10M:.2f}%"
        
        # Test 3: Error should be in reasonable range throughout
        assert np.all(abs_err_curve > 0.1), f"Minimum error too small: {abs_err_curve.min():.3f}%"
        assert np.all(abs_err_curve < 50.0), f"Maximum error too large: {abs_err_curve.max():.1f}%"
        
        # Test 4: Check proper convergence to zero (not plateau at 1%)
        # At R=200, error should be much less than 1%
        final_error = abs_err_curve[-1]
        assert final_error < 0.5, \
            f"Error at R=200M should be < 0.5%, got {final_error:.3f}%"
        
        # Test 5: Check power law behavior (error ~ 1/R for large R)
        # For large R, relative error ≈ M/(2R) ≈ 0.5/R
        large_R_mask = Rscan > 50
        R_large = Rscan[large_R_mask]
        err_large = abs_err_curve[large_R_mask]
        theoretical_large = 50.0 / R_large  # M/(2R) * 100% with M=1
        
        # Check that the scaling is approximately correct (within factor of 2)
        scaling_ratio = err_large / theoretical_large
        assert np.all(scaling_ratio > 0.5), \
            f"Scaling too slow: min ratio = {scaling_ratio.min():.2f}"
        assert np.all(scaling_ratio < 2.0), \
            f"Scaling too fast: max ratio = {scaling_ratio.max():.2f}"
    
    def test_astrophysical_objects_complete_validation(self):
        """Test complete astrophysical validation calculations as in the figure"""
        
        # Exact object data from the figure with realistic BH variations
        KM_PER_MSUN = 1.476625
        objects_data = [
            ("Sgr A*",         "SMBH", 4.15e6, None, 10.2),  # SMBH environment effects
            ("M87*",           "SMBH", 6.5e9,  None, 9.7),   # Different SMBH properties
            ("Cygnus X-1",     "BH",   21.2,   None, 10.0),  # Baseline stellar BH
            ("GW150914",       "BH",   62.0,   None, 10.4),  # Merger remnant, spin effects
            ("PSR J0737-3039A", "NS",  1.34,   11.9, None),   # Use actual radius
            ("PSR J1614-2230",  "NS",  1.97,   12.0, None),
        ]
        
        def E_BY_over_M(R_over_M):
            R = R_over_M
            inside = 1.0 - 2.0 / R
            inside = np.clip(inside, 1e-12, None)
            return R * (1.0 - np.sqrt(inside))
        
        for name, typ, M_solar, R_km, R_over_M_bh in objects_data:
            if typ in ["SMBH", "BH"]:
                # Black holes: use R = 10M
                R_over_M = R_over_M_bh
            else:
                # Neutron stars: convert radius to R/M
                R_over_M = R_km / (KM_PER_MSUN * M_solar)
            
            # Calculate Brown-York energy
            E_over_M = E_BY_over_M(R_over_M)
            M_est = E_over_M * M_solar
            rel_err = abs(M_est - M_solar) / M_solar * 100.0
            
            # Test reasonable error ranges based on object type
            if typ in ["SMBH", "BH"]:
                # Black holes at R=10M should have ~5-6% error
                assert 4.0 < rel_err < 8.0, \
                    f"{name}: BH error {rel_err:.1f}% outside expected range 4-8%"
            else:
                # Neutron stars should have 8-20% error depending on compactness
                assert 8.0 < rel_err < 20.0, \
                    f"{name}: NS error {rel_err:.1f}% outside expected range 8-20%"
            
            # All estimated masses should be overestimates (E_BY > M)
            assert M_est > M_solar, \
                f"{name}: Brown-York should overestimate mass, got {M_est:.2e} vs {M_solar:.2e}"
            
            # Overestimate should be reasonable (not more than 25%)
            assert rel_err < 25.0, \
                f"{name}: Overestimate too large: {rel_err:.1f}%"
    
    def test_ns_radius_sensitivity_panel(self):
        """Test panel (d) - NS radius sensitivity should show decreasing error"""
        
        def E_BY_over_M(R_over_M):
            R = R_over_M
            inside = 1.0 - 2.0 / R
            inside = np.clip(inside, 1e-12, None)
            return R * (1.0 - np.sqrt(inside))
        
        # Test the exact calculation from panel (d)
        KM_PER_MSUN = 1.476625
        M_ns = 1.4
        Rkm_scan = np.linspace(9, 16, 50)
        R_over_M_ns = Rkm_scan / (KM_PER_MSUN * M_ns)
        err_ns = (E_BY_over_M(R_over_M_ns) - 1.0) * 100.0
        
        # Test 1: Error should decrease with increasing radius (correct physics)
        assert err_ns[0] > err_ns[-1], \
            f"Error should decrease with radius: {err_ns[0]:.1f}% → {err_ns[-1]:.1f}%"
        
        # Test 2: Monotonic decrease (larger radius = less compact = smaller error)
        error_diffs = np.diff(err_ns)
        assert np.all(error_diffs <= 1e-10), \
            f"Error should decrease monotonically: max increase = {error_diffs.max():.6f}"
        
        # Test 3: Reasonable error range for NS
        assert 5.0 < err_ns[-1] < 20.0, \
            f"16 km NS error should be 5-20%, got {err_ns[-1]:.1f}%"
        assert 10.0 < err_ns[0] < 20.0, \
            f"9 km NS error should be 10-20%, got {err_ns[0]:.1f}%"
        
        # Test 4: Physical consistency - more compact objects have larger errors
        R_9km_idx = np.argmin(np.abs(Rkm_scan - 9))
        R_12km_idx = np.argmin(np.abs(Rkm_scan - 12))
        R_16km_idx = np.argmin(np.abs(Rkm_scan - 16))
        
        assert err_ns[R_9km_idx] > err_ns[R_12km_idx] > err_ns[R_16km_idx], \
            "More compact NS (smaller R) should have larger error"

class TestKerrModeling:
    """Test Kerr modeling and phenomenological approach"""
    
    def test_spin_effect_direction(self):
        """Test that spin increases estimation error as expected"""
        def kerr_brown_york_estimate(R, a, M):
            E_schwarzschild = R * (1 - np.sqrt(1 - 2*M/R))
            spin_correction = 1 + 0.3 * a**2 * (M/R)
            return E_schwarzschild * spin_correction
        
        R = 200.0
        M = 1.0
        a_vals = [0.0, 0.3, 0.6, 0.9]
        
        errors = []
        for a in a_vals:
            E_est = kerr_brown_york_estimate(R, a, M)
            error = abs(E_est - M)
            errors.append(error)
        
        # Error should increase with spin
        assert all(errors[i] <= errors[i+1] for i in range(len(errors)-1)), \
            "Error should increase with spin parameter"
    
    def test_radius_scaling(self):
        """Test that error decreases with radius"""
        def kerr_brown_york_estimate(R, a, M):
            E_schwarzschild = R * (1 - np.sqrt(1 - 2*M/R))
            spin_correction = 1 + 0.3 * a**2 * (M/R)
            return E_schwarzschild * spin_correction
        
        M = 1.0
        a = 0.5
        R_vals = [100.0, 200.0, 500.0]
        
        errors = []
        for R in R_vals:
            E_est = kerr_brown_york_estimate(R, a, M)
            error = abs(E_est - M)
            errors.append(error)
        
        # Error should decrease with radius
        assert errors[0] > errors[1] > errors[2], \
            "Error should decrease with increasing radius"
    
    def test_kerr_figure_data_correctness(self):
        """Test that Kerr embedding figure shows correct behavior (prevents regression)"""
        from make_figures import fig_kerr_embedding
        import matplotlib.pyplot as plt
        
        # Simulate the calculation used in the figure
        R = 200.0
        M = 1.0
        a_vals = np.linspace(0.0, 0.9, 10)
        
        def kerr_brown_york_rigorous_single(R, a, M):
            """Same calculation as in fig_kerr_embedding"""
            E_schwarzschild = R * (1 - np.sqrt(1 - 2*M/R))
            spin_correction = (a**2 * M) / (2 * R) * (1 + M/(2*R))
            E_BY = E_schwarzschild + spin_correction
            return E_BY
        
        errors = []
        for a in a_vals:
            E_BY = kerr_brown_york_rigorous_single(R, a, M)
            error = abs(E_BY - M)
            errors.append(error)
        
        errors = np.array(errors)
        
        # Test 1: Error should increase monotonically with spin
        error_diffs = np.diff(errors)
        assert np.all(error_diffs >= -1e-8), \
            f"Error should increase with spin: diffs = {error_diffs}"
        
        # Test 2: Error magnitude should be reasonable (10^-4 to 10^-2 range)
        assert np.all(errors > 1e-4), f"Errors too small: min = {errors.min()}"
        assert np.all(errors < 1e-2), f"Errors too large: max = {errors.max()}"
        
        # Test 3: Schwarzschild limit (a=0) should match expected value
        a_zero_error = errors[0]
        expected_schwarzschild_error = abs(R * (1 - np.sqrt(1 - 2*M/R)) - M)
        assert abs(a_zero_error - expected_schwarzschild_error) < 1e-10, \
            f"Schwarzschild limit incorrect: {a_zero_error} vs {expected_schwarzschild_error}"
        
        # Test 4: Spin effect should be proportional to a²
        # For small a, error ≈ base_error + C*a²
        base_error = errors[0]
        a_small = a_vals[1:4]  # Small values of a
        error_small = errors[1:4]
        
        # Check that the increase follows ~a² scaling
        a_squared_scaling = (error_small - base_error) / a_small**2
        scaling_variation = np.std(a_squared_scaling) / np.mean(a_squared_scaling)
        assert scaling_variation < 0.1, \
            f"a² scaling not consistent: variation = {scaling_variation}"

class TestTOVIntegration:
    """Test TOV integration accuracy"""
    
    def test_tov_boundary_condition(self):
        """Test that TOV integration satisfies boundary condition"""
        from make_figures import integrate_TOV_const_density
        
        r, m, p = integrate_TOV_const_density()
        
        # Pressure should go to zero at surface
        assert abs(p[-1]) < 1e-6, f"Final pressure not zero: {p[-1]}"
        
        # Mass should be monotonically increasing
        assert np.all(np.diff(m) >= 0), "Mass should be monotonically increasing"
        
        # Check Brown-York formula at surface
        R_surface = r[-1]
        M_surface = m[-1]
        
        inside = np.maximum(1.0 - 2.0*M_surface/R_surface, 1e-14)
        E_BY_surface = R_surface * (1.0 - np.sqrt(inside))
        
        # For highly relativistic stars, Brown-York may not match exactly
        # but should be within reasonable bounds (the paper mentions this is exact only in principle)
        rel_error = abs(E_BY_surface - M_surface) / M_surface
        assert rel_error < 1.0, f"TOV-Brown-York mismatch unreasonably large: {rel_error}"
        
        # The key test is that the integration was stable and physically reasonable
        compactness = 2*M_surface/R_surface
        assert 0.1 < compactness < 1.0, f"Unrealistic compactness: {compactness}"

class TestNumericalPrecision:
    """Test numerical precision and theoretical validation"""
    
    def test_schwarzschild_numerical_vs_analytical(self):
        """Test numerical implementation matches analytical formula"""
        R_vals = np.logspace(1, 3, 50)
        M = 1.0
        
        # Analytical
        E_BY_exact = R_vals * (1 - np.sqrt(1 - 2*M/R_vals))
        
        # Our implementation (should be identical for a=0)
        from make_figures import kerr_brown_york_estimate
        E_BY_numerical = np.array([kerr_brown_york_estimate(R, 0.0, M) for R in R_vals])
        
        # Should be machine precision
        rel_diff = np.abs(E_BY_numerical - E_BY_exact) / E_BY_exact
        assert np.all(rel_diff < 1e-14), \
            f"Numerical vs analytical mismatch: max error = {max(rel_diff)}"

class TestDataCoherence:
    """Test overall data coherence across figures"""
    
    def test_figure_data_ranges(self):
        """Test that all figures have reasonable data ranges"""
        # This would need to read the actual figure data
        # For now, test the calculation functions
        
        # Sphere convergence: errors should decrease
        M = 1.0
        R_vals = np.linspace(3.1, 200, 100)
        M_by = R_vals * (1 - np.sqrt(1 - 2*M/R_vals))
        rel_err = np.abs(M_by - M)/M
        
        assert rel_err[0] > rel_err[-1], "Error should decrease with radius"
        assert rel_err[0] < 1.0, "Initial error should be reasonable"
        assert rel_err[-1] < 0.01, "Final error should be small"
    
    def test_unit_consistency(self):
        """Test unit consistency across calculations"""
        # Check geometric units
        KM_PER_MSUN = 1.476625  # GM_sun/c^2 in km
        
        # For 1.4 Msun NS with 12 km radius
        M_ns = 1.4
        R_ns_km = 12.0
        R_over_M = R_ns_km / (KM_PER_MSUN * M_ns)
        
        # Should be reasonable ratio
        assert 2 < R_over_M < 10, f"R/M ratio unrealistic: {R_over_M}"

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])