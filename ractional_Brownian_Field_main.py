from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
import math
from tqdm import tqdm
from mpmath import mp, mpf, gamma
from scipy.special import lambertw

# Reference: Peng and Wu (2025), with the python package fractal-analysis 0.3.5, 
# and the link is https://pypi.org/project/fractal-analysis/

class WoodChanFgnSimulator:
    def __init__(self, sample_size: int, hurst_parameter: float, tmax: float = 1):
        self.sample_size = sample_size
        if self.sample_size <= 0:
            raise ValueError(f'sample_size must be a positive integer.')
        self.hurst_parameter = hurst_parameter
        if not (0 < self.hurst_parameter < 1):
            raise ValueError(f'hurst_parameter must be in range (0, 1).')
        self.tmax = tmax
        if not self.tmax > 0:
            raise ValueError(f'tmax must be positive.')

    def _first_line_circulant_matrix(self, m, cov: Callable, prev_k=None, prev_v=None):
        new_k = self.tmax * np.arange(0, m / 2 + 1, dtype=int)

        if prev_k is not None and prev_v is not None:
            # Reuse previous computed values
            prev_len = len(prev_k)
            if prev_len >= len(new_k):
                v = prev_v[:len(new_k)]
            else:
                extra_k = new_k[prev_len:]
                extra_v = cov(k=extra_k)
                v = np.concatenate((prev_v, extra_v))
        else:
            v = cov(k=new_k)

        ind = np.concatenate((np.arange(0, m / 2, dtype=int), np.arange(m / 2, 0, -1, dtype=int)))
        line = v[ind]
        return line, new_k, v

    @staticmethod
    def _simulate_w(m, seed: int = None):
        np.random.seed(seed)
        ar = np.random.normal(0, 1, int(m / 2 + 1))
        ai = np.random.normal(0, 1, int(m / 2 + 1))
        ar[0] = 2 ** 0.5 * ar[0]
        ar[-1] = 2 ** 0.5 * ar[-1]
        ai[0] = 0
        ai[-1] = 0
        ar = np.concatenate((ar, ar[int(m / 2 - 1): 0:-1]))
        aic = -ai
        ai = np.concatenate((ai, aic[int(m / 2 - 1): 0:-1]))
        w = [complex(one_ar, one_ai) for one_ar, one_ai in zip(ar, ai)]
        return w

    def get_fgn(self, cov: Callable, N: int, seed: int = None, is_precise: bool = False) -> np.ndarray:
        m = 2 ** (int(math.log(N - 1) / math.log(2) + 1))
        eigc, k_vals, v_vals = self._first_line_circulant_matrix(m=m, cov=cov)
        eigc = fft(eigc)
        if not is_precise:
            eigc = np.clip(eigc, 1e-10, None)
        else:
            while any(v <= 0 for v in eigc) and m < 2 ** 17:
                m = 2 * m
                eigc, k_vals, v_vals = self._first_line_circulant_matrix(m=m, cov=cov, prev_k=k_vals, prev_v=v_vals)
                eigc = fft(eigc).real
        w = self._simulate_w(m=m, seed=seed)
        # reconstruction of the fgn
        w = np.sqrt(eigc.astype(np.cdouble)) * w
        fgn = fft(w)
        fgn = fgn / (2 * m) ** 0.5
        fgn = fgn.real
        return fgn

class RandomFieldSelfSimilarFractalSimulator(WoodChanFgnSimulator):
    def __init__(self, sample_size: int, hurst_parameter: float, covariance_func: Callable, initial_value = float, factor: float = None, tmax: float = 1):
        sample_size = int(sample_size)
        self.dpw_size = sample_size
        fgn_size = np.ceil(np.real(lambertw(-np.log(1/(sample_size-1)+1), k=-1))/(-np.log(1/(sample_size-1)+1)))
        self.fgn_size = fgn_size.astype('int')
        super().__init__(sample_size=self.fgn_size, hurst_parameter=hurst_parameter, tmax=tmax)
        self.covariance_func = covariance_func
        self.factor = factor
        self.initial_value = initial_value

    @property
    def _lamperti_subseq_index(self):
        seires_step = self.tmax / self.dpw_size
        series_t = np.arange(start=seires_step, stop=self.tmax + seires_step, step=seires_step)
        # shifting negative time index to positive time index
        log_series_t = np.log(series_t) + np.abs(np.log(series_t[0]))
        max_log_series_exp_t = np.max(log_series_t)
        lamperti_subseq_index = np.rint(log_series_t * self.fgn_size / max_log_series_exp_t) - 1
        lamperti_subseq_index[0] = 0
        return lamperti_subseq_index.astype(int)

    def get_self_similar_process(self, seed=None):
        seires_step = self.tmax / self.dpw_size
        series_t = np.arange(start=seires_step, stop=self.tmax + seires_step, step=seires_step)
        lamp_fgn = self.get_fgn(seed=seed, N=self.fgn_size, cov=self.covariance_line)[:-1]
        lamp_fgn = np.cumsum(lamp_fgn)
        lamp_fbm = series_t ** (self.hurst_parameter * self.factor) * lamp_fgn[self._lamperti_subseq_index]
        new_fgn = lamp_fbm[1:] - lamp_fbm[:-1]
        new_fgn = np.concatenate([[self.initial_value], new_fgn])
        return np.cumsum(new_fgn)

    def covariance_with_adaptive_precision(self, k_de, n_de, hurst_de, factor_de, tolerance=0.0001, initial_prec=17,
                                           step=5, max_prec=1000000):
        mp.dps = initial_prec
        tolerance = mpf(str(tolerance))
        v = self.covariance_func(k_de=k_de, n_de=n_de, hurst_de=hurst_de, factor_de=factor_de)
        return float(v)
        
    def covariance_line(self, k):
        n_de = mpf(str(self.fgn_size))
        hurst_de = mpf(str(self.hurst_parameter))
        if self.factor is not None:
            factor_de = mpf(str(self.factor))
        else:
            factor_de = None
        v = np.array([
            self.covariance_with_adaptive_precision(k_de=mpf(str(k_ele)), n_de=n_de, hurst_de=hurst_de,
                                                    factor_de=factor_de) for k_ele in k])
        return v

class RandomFieldSimulator(RandomFieldSelfSimilarFractalSimulator):
    def __init__(self, sample_size: int, hurst_parameter: float, initial_value: float,
                 tmax: float = 1, FBM_cov_md: int = 1, rf_factor: float=0.7):

        self.rf_factor = rf_factor
        if FBM_cov_md == 1:
            super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter,
                             initial_value = initial_value, covariance_func=self.fbm_cov, 
                             factor=1, tmax=tmax)
        elif FBM_cov_md == 2:
            super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter,
                            initial_value = initial_value, covariance_func=self.sub_fbm_cov, 
                            factor=1, tmax=tmax)
        elif FBM_cov_md == 3:
            super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter,
                            initial_value = initial_value, covariance_func=self.bi_fbm_cov, 
                            factor=self.rf_factor, tmax=tmax)
        elif FBM_cov_md == 4:
            super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter,
                            initial_value = initial_value, covariance_func=self.tri_fbm_cov, 
                            factor=self.rf_factor, tmax=tmax)
    
    def fbm_cov(self, k_de, n_de, hurst_de, factor_de):
        N = n_de
        t_s = k_de/N
        s_t = -t_s
        H = hurst_de

        # Abs part
        temp_1 = np.abs(N**(t_s/2) - N**(s_t/2))**(2*H)
        temp_2 = np.abs(N**((t_s-1/N)/2) - N**((s_t+1/N)/2))**(2*H)
        temp_3 = np.abs(N**((t_s+1/N)/2) - N**((s_t-1/N)/2))**(2*H)
        abs_res = - temp_1 + (temp_2 + temp_3)/2
        
        #No abs part
        new_temp_1 = N**(t_s*H) + N**(s_t*H)
        new_temp_2 = N**((t_s-1/N)*H) + N**((s_t+1/N)*H)
        new_temp_3 = N**((t_s+1/N)*H) + N**((s_t-1/N)*H)
        no_abs_res = new_temp_1 - (new_temp_2 + new_temp_3)/2
        return abs_res + no_abs_res
    
    def C_H(self, H):
        if np.abs(H - 0.5) < 0.001:
            return np.pi
        else:
            return gamma(2-2*H)*np.cos(np.pi*H) / (H*(1-2*H))
    

    
