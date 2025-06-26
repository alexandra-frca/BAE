'''
Algorithms for sampling from the distributions in approximate Bayesian 
inference, namely sequential Monte Carlo with the Liu-West filter (SMC-LW)
and with Markov chain Monte Carlo (MCMC): random walk Metropolis (RWM).
'''
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

from src.utils.models import LikelihoodModel
from src.utils.mydataclasses import MeasurementData
from src.utils.misc import (get_truncated_normal, k_smallest_tuples, print_1st_err,
                        logsumexp, print_centered, np_invalid_catch, 
                        k_largest_tuples, lprint, PrintSeqTable)
rng = np.random.default_rng()
#plt.rcParams['axes.formatter.useoffset'] = False
first_init = True
first_conditional_utility = True

INIT = {var: False for var in ["a", "theta", "Tc"]}

class SMCsampler():
    def __init__(self, model: LikelihoodModel, Npart = 1000,  thr = 0.5, 
                 var="a", ut="var", Tc_est = None, plot = False, log = False, 
                 prior = "uniform", res_ut = False):
        assert var in ["a", "theta", "Tc"]
        assert ut in ["var", "varN2", "ESS"]
        assert prior in ["uniform", "normal"]

        self.Npart = Npart
        # self.logl_catch(np.array([0]))
        self.model = model
        self.range = model.get_range(var)

        self.var = var 
        self.ut = ut 
        self.tgESS = thr
        if self.ut == "ESS":
            print("> Resampling threshold to be set to 1 for single shot "
                  f"measurements due to ESS-based utility; target ESS = {thr}.")
        self.plot = plot
        # log: only for resampling. Weights are always log.
        self.log = log
        # og: whether the sampler is stand-alone, or was created for auxiliary 
        # calculations.
        self.og = True
        
        
        self.init_prior(prior)
        
        # Whether to resample in the utility calculations.
        self.res_ut = res_ut
        
        # Number of single shot measurements considered.
        self.Ndata = 0
        self.Nupdates = 0
        self.resampler_calls = 0
        self.str = None
        self.ess_list = []
        self.resampled_list = []
        
        # self.first_underflow = True

        
        if INIT[var]:
            print("> Initialized a sampler with:")
            info = [f"Nparticles = {Npart} | tgESS = {thr}", 
                    f"internal variable: {var} (range = {self.range})",
                    f"utility function: {ut}"]
            print_centered(info)
            INIT[var] = False

    def init_prior(self, dist):
        if dist == "uniform":
            self.locs = np.random.uniform(*self.range, size = self.Npart)
        if dist == "normal":
            mn, mx = self.range
            Tnorm = get_truncated_normal(0.5, 1, low=mn, upp=mx)
            self.locs = Tnorm.rvs(size = self.Npart)
        self.set_unif_weights()
        
    def set_unif_weights(self):
        # Do not use Npart; causes issues if resampling removes particles.
        N = len(self.locs)
        self.ws = np.log(np.ones(N))
        # Save sums for variance and ESS calculations.
        self.lnorm = np.log(N)
        self.lnorm2 = np.log(N)
        
    def batch_update(self, data: MeasurementData):
        self.Nupdates += 1
        batch_likelihood = self.model.batch_likelihood
        param_and_lls = [batch_likelihood(loc, data) for loc in self.locs]
        self.locs = [x[0] for x in param_and_lls]
        self.ws = [x[1] + w for x, w in zip(param_and_lls, self.ws)]
        
    def update_seq(self, data):
        '''
        Update particle cloud considering multiple data, but one at a time to
        allow for resampling in between.
        '''
        N = len(data)
        for i in range(N):
            self.update_latest(data.partial_data(i+1))
            self.plots(data.partial_data(i+1))
        self.Nupdates += N
        
    @property
    def rthr(self):
        '''
        Resampling threshold. 
        
        If res_ut is False and this is an auxiliary sampler for the utility
        calculations (self.og is False), do not resample.
        
        If using a ESS based utility, measurements are chosen to achieve target 
        ESS, so resample every time (don't want to go below the target). 
        '''
        if not (self.og or self.res_ut):
            return 0
        if self.ut == "var":
            return self.tgESS
        if self.ut == "ESS":
            return 1
    
    def update_latest(self, data):
        '''
        Update based on yet not considered data.
        '''
        for j in range(self.Ndata, len(data)):
            self.update_aux(data.partial_data(j + 1))
            self.Ndata += 1
        # Plot only after all shots. 
        self.plots(data)
        
    def update_aux(self, data):
        '''
        Update for a single shot.
        '''
        # Get the latest datum for the weight updates.
        ctrl, outcome, nshots = (data.ctrls[-1], data.outcomes[-1], data.Nsshots[-1])
        # Get the likelihoods associated with each particle.
        likelihood = self.model.likelihood
        ls = likelihood(self.locs, ctrl, outcome, nshots, var = self.var)
        
        # Reweight. Separate steps as if likelihood is 0, logl_catch removes
        # particles, shortening the list of weights.
        lls = self.logl_catch(ls)
        self.ws += lls

        # Update normalization/ESS info.
        self.calc_sums()
            
        if self.whether_resample():  
            self.resample(data)
            self.resampler_calls += 1
            self.resampled_list.append(self.Nupdates)
            
        self.Nupdates += 1
        
    def calc_sums(self):
        '''
        Calculate sum of weights and sum of squared weights (exponentiate 
        back from log).
        '''
        # Logarithm of the sum.
        self.lnorm = logsumexp(self.ws)
        # Logarithm of the sum of squares. 
        self.lnorm2 = logsumexp(self.ws*2)
        
        # self.wsum = np.sum(np.exp(self.ws))
        # self.wsum2 = np.sum(np.exp(self.ws)**2)
    
    def whether_resample(self):
        # Resample if the weights are dominated by too small a fraction of
        # the particles (statistical power is critically low).
        ESS = self.ESS()
        ESS_thr = self.rthr*self.Npart
        if ESS < ESS_thr:
            return True
        else:
            return False
        
    def ESS(self, append = True):
        ESS = np.exp(2*self.lnorm - self.lnorm2)
        '''
        if self.og:
            print("> ESS difference to target: ", 
                  abs((ESS/self.Npart)-self.tgESS))
        '''
        if append:
            self.ess_list.append(ESS)
        return ESS
        
    def logl_catch(self, ls, print_which = True):
        def cleanup(ls):
            zeros_pc = 100*np.sum(ls == 0)/len(ls)
            print(f"> {zeros_pc:.1f}% of likelihoods were zero.")
            
            if print_which:
                print("> For: ")
                # Nonzero gets indices. Returns a tuple for the dimensions.
                for i in np.nonzero(ls == 0)[0]:
                    loc, w = self.locs[i], self.get_rw(i)
                    print(f"{i = }, {loc = }, {w = }", end = " | ")
                print("(removed particles)")
                # input()

            if np.isclose(zeros_pc, 100):
                # Pause execution and give opportunity to keyboard interrupt.
                input()

            self.locs = self.locs[ls != 0]
            self.ws = self.ws[ls != 0]
            ls = ls[ls != 0]
            lls = np.log(ls)
            return lls
        
        assert all(ls >= 0)
        lls = np_invalid_catch(f = np.log,
                             largs = [ls], 
                             errdesc = ["divide by zero encountered in log"], 
                             cleanup = cleanup,
                             caller = "logl_catch")
        return lls
        
    def resample(self, data, testat = np.inf):
        '''
        Change particle locations to introduce variability. The mechanism may
        vary. There is a trade-off between statisticall correctness, processing
        cost, and space exploration.

        Test at: test the resampling for a specific dataset length, by applying
        it multiple times. 
        '''
        
        if len(data) > testat:
            self.test_resampling(data)
            
        # Multinomial sampling with replacement from the current particle cloud.
        self.locs = rng.choice(self.locs, size=self.Npart, p=self.rws)
        self.set_unif_weights()
        # Introduce variability by resampling.
        self.resampler(data)
        
    def test_resampling(self, data, steps = 100, every = 100):
        print("> Testing resampling...")
        self.model.plot_likelihood(data, ttl_xtra = " (resampling test)")
        self.model.mle(data)
        self.model.mean_estimator(data)
        self.plot_particles(ttl_xtra = "- initial (resampling test)")
        
        locsample = self.locs_evol()
        st = PrintSeqTable(["mean", "std", "example loc", "example a"])
        st.print_row(self.test_list())
        input()

        for i in range(steps):
            self.resample(data, testat = np.inf)
            locsample = self.locs_evol(locsample = locsample)
            if i % every == 0:
                self.plot_particles(ttl_xtra = f"- after step {i} (resampling test)")
                st.print_row(self.test_list())
                input()
            
    def locs_evol(self, locsample = None, slen = 3, every = 20):
        '''
        Follow evolution of slen random particles (order irrelevant so can be 
        first). After 'every' evols, print mean for each and start over.
        
        This only makes sense when resampling continuously, otherwise the 
        particles are not the same. Made for testing the resampling.
        '''
        if locsample is None:
            # Create array [[loc1], [loc2],...]
            locsample = self.locs[:slen].reshape(-1,1)
            print(f"> First {slen} locs: ", *locsample.flatten())
            return locsample
            
        locsample = np.hstack((locsample, self.locs[:slen].reshape(-1,1)))
        
        # print(locsample, locsample.shape)
        if locsample.shape[1] % every == 0:
            means_all = np.mean(locsample, axis = 1)
            latest = np.array([l[:every] for l in locsample])
            means_latest = np.mean(latest, axis = 1)
            print(f"> Means over all {locsample.shape[1]} steps:    ", *means_all)
            print(f"> Means over {every} latest steps: ", *means_latest)
            
        return locsample
        
    def test_list(self):
        '''
        Return a list with the mean, standard deviation, an example particle,
        and the parameter of interesting corresponding to the example particle.
        '''
        return [*self.mean_and_std(), self.locs[0], self.param_of_interest[0]]
    
    @property
    def param_of_interest(self):
        if self.var in ["a", "Tc"]:
            param = self.locs
        elif self.var=="theta":
            param = self.model.a_from_theta(self.locs)
        else:
            raise Exception("`var`must be 'a' or 'theta' or 'Tc'.")
        return param
    
    
    def mean_and_std(self):
        mean = self.mean()
        var = self.variance(mean)

        if var == 0:
            print(f"> Variance is zero. Mean is {mean}.")
            self.print_lists()
        return mean, np.sqrt(var)

    def variance(self, mean = None):
        if mean is None:
            mean = self.mean()
        var = self.variance_catch(mean)
        return var
    
    def variance_catch(self, mean):
        def cleanup(xs, ws):
            rws = self.rws
            xws = list(zip(xs,rws))
            print(f"> Overflow in variance calculation; mean is {mean}.")
            print(f"> lnorm = {self.lnorm}")
            print("> (x,w) with largest x:", k_largest_tuples(xws, 3))
            f = lambda x: (x-mean)**2
            print_1st_err(f, xs, "x", ls = [rws], strs = ["w"])
        
        var_calc = lambda xs, ws: np.sum(np.multiply(np.square(xs-mean),
                                                np.exp(ws - self.lnorm)))
        r = np_invalid_catch(f = var_calc,
                             largs = [self.param_of_interest, self.ws], 
                             errdesc = ["overflow encountered in square"], 
                             cleanup = cleanup,
                             caller = "variance_catch")
        return r
        
    def mean(self):
        # mean = np.average(self.locs, weights=self.ws)
        mean = np.sum(self.param_of_interest*np.exp(self.ws - self.lnorm))
        if mean == 0:
            print("> Mean is zero.")
            self.print_lists()
            input()
        return mean
    
    def normalize(self, quantity):
        return self.normalize_catch(quantity)
    
    def normalize_catch(self, quantity):
        def cleanup(quantity):
            print("> Overflow when normalizing quantity.", end = " ")
            print(f"Unnormalized value = {quantity}, lnorm = {self.lnorm}.")
            
        nval_calc = lambda x: np.exp(np.log(x)-self.lnorm) 
        r = np_invalid_catch(f = nval_calc,
                             largs = [quantity], 
                             errdesc = "overflow encountered in exp", 
                             cleanup = cleanup,
                             caller = "normalize_catch")
        return r
    
    @property
    def rws(self):
        '''
        Return (non log, normalized) weights. 
        '''
        return np.exp(self.ws - self.lnorm)
        
    def get_rw(self, i):
        '''
        Return real (non log, normalized) weight for index i.
        '''
        return np.exp(self.ws[i]-self.lnorm)
    
    def expected_utilities(self, ctrls: [float], data):
        utils = [self.expected_utility(ctrl, data) for ctrl in ctrls]
        # utils = self.expected_utility(ctrls, data) 
        return np.array(utils)
    
    def expected_utility(self, ctrl: float, data):
        util = 0
        outcomes = [0,1]
        for outcome in outcomes:
             # Calculate the expected probability of 'outcome'.
            p = self.expected_probability(ctrl, outcome)    
            # Calculate the conditional utility given 'outcome'.
            if not math.isclose(p,0):
                cutil = self.conditional_utility(ctrl, outcome, data)
                util += p*cutil
        return util
    
    def conditional_utility(self, ctrl, outcome, data):
        # Make copies to preserve the actual dataset and distribution. 
        
        data_cpy = deepcopy(data)
        data_cpy.prt = False
        data_cpy.append_datum(ctrl, outcome, 1)
        
        sampler_cpy = deepcopy(self)
        sampler_cpy.plot = False
        sampler_cpy.og = False
        sampler_cpy.update_latest(data_cpy)
        
        if self.ut == "var":
            _, std = sampler_cpy.mean_and_std()
            var = std**2
            return -var
        elif self.ut == "varN2":
            _, std = sampler_cpy.mean_and_std()
            var = std**2*ctrl**2
            return -var
        elif self.ut == "ESS":
            ESS = sampler_cpy.ess_list[-1]
            return -abs(self.tgESS-ESS/self.Npart)

            
    def expected_probability(self, ctrl, outcome):
        likelihood = self.model.likelihood
        ps = np.multiply(np.exp(self.ws - self.lnorm), likelihood(self.locs, ctrl, outcome, 1,
                                                  var = self.var))
        p = np.sum(ps)
        return p
    
    def plots(self, data, ttl_xtra = ""):
        if self.plot:
            self.model.plot_likelihood(data, ttl_xtra=f"{ttl_xtra}, {self.var}")
            self.plot_particles(ttl_xtra = ttl_xtra, lendata = len(data))
    
    def plot_particles(self, ax = None, lendata = None, ttl_xtra = ""):
        if ax is None:
            fig, ax = plt.subplots(1,figsize=(10,6))
            
        ax.scatter(self.param_of_interest, np.exp(self.ws), s = 20, 
                   facecolors="darkgray", color = "dimgray",
                   label = "SMC particles")
        plt.legend(loc="upper right", fontsize=14, framealpha=0.8)
        
        if ax is None:
            return
        
        datainfo = f", {lendata} data" if lendata is not None else ""
        ax.set_title(f"SMC particle cloud #{self.Nupdates} {datainfo}"
                     f" {ttl_xtra}", fontsize=16, pad=25)
        ax.set_xlabel("Particle location (a)", fontsize=16, style="italic", 
                      labelpad=10)
        ax.set_ylabel("Particle weight", fontsize=16, style="italic", 
                      labelpad=10)
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        ax.xaxis.set_major_locator(MaxNLocator(10))
        ax.yaxis.set_major_locator(MaxNLocator(7))
        
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        
        ax.spines['right'].set_color('lightgray')
        ax.spines['top'].set_color('lightgray')
        
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2, linestyle='--')
        ax.grid(which='major', alpha=0.8)
        plt.show()
        
    def print_stats(self):
        if self.Nupdates == 0:
            print(" > Sampler initialized but not used.")
            return
        Nr = self.resampler_calls
        Nr_pcent = int(round(100*Nr/self.Nupdates))
        # print("> Used ESS based utility")
        print(f"Total resampler calls: {Nr} ({Nr_pcent}%) [SMCsampler]")
        ess_list = np.round(np.array(self.ess_list))
        print(f"> Mean ESS: {np.mean(ess_list)}")
        ess_list = list(enumerate(ess_list))
        # print("> ESS after each single-shot update (immediately after re-weighting): ", 
        #       ess_list)
        print("> Resampled at measurements (single shot): ", self.resampled_list)
        # Remove 'np.float64' part for printing.
        floatlist = [(i,float(ess)) for i,ess in ess_list] 
        print("> 5 smallest registered ESS: ", k_smallest_tuples(floatlist, 5, sortby=1))
        
    
class LiuWestSampler(SMCsampler):
    
    def __init__(self, model, a_LW = 0.98, **kwargs):
        super().__init__(model, **kwargs)
        self.a_LW = a_LW
        self.str = "LW"
        
        global first_init
        if first_init:
            print(f"> Resampler: Liu-West (a = {self.a_LW})."
                      " [LiuWestSampler]")
            first_init = False
            
    def resampler(self, data, truncated = False):
        '''
        The data are not used but we need them in the signature.
        
        truncated: whether to use a truncated normal distribution considering
        the domain, or to just use a standardnormal distribution.
        '''
        currmean, currstd = self.mean_and_std()
        a = self.a_LW
        means = a*self.locs+(1-a)*currmean
        h = (1-a**2)**0.5
        std = h*currstd
        
        if truncated:
            mn, mx = self.range
            Tnorm = get_truncated_normal(means, std, low=mn, upp=mx)
            new_locs = Tnorm.rvs()
        else:
            new_locs = np.random.normal(means, std)
        return new_locs

class MetropolisSampler(SMCsampler):
    
    def __init__(self, model, c=2.38, **kwargs):
        super().__init__(model, **kwargs)
        
        # Initialize 2 counters for the average acceptance ratio = acc_r/Msteps
        self.acc_r = 0
        self.Msteps = 0
        self.c = c
        self.str = "RWM"    
        self.acc_rates = []
        global first_init
        if first_init:
            print(f"> Resampler: RW Metropolis (c={c}). "
                  "[MetropolisSampler]")
            first_init = False
        
    def resampler(self, data):
        # Propose new locations.
        proposals = self.generate_proposals(data)
        # Calculate acceptance rates.
        acc_rates, proposals = self.acceptance_rates(proposals, data)

        # Probabilistic rejection step.
        self.rejection_step(acc_rates, proposals)

        '''
        if self.acc_rates[-1] < 0.1 and self.og:
            print("> MCMC acceptance rate < 0.1", self.acc_rates)
            if all(np.array(self.acc_rates)[-5:] < 0.15):
                print("> 5 latests acc rates < 0.15!")
        '''

        
    
    def generate_proposals(self, data, truncated = False):
        '''
        Propose new locations via a Gaussian perturbation of the old ones.

        The variance of the Gaussian is chosen to be proportional to that of the
        current SMC distribution, which approximates the target distribution.
        '''
        # Get the parameter for the proposal distribution.
        _, currsd = self.mean_and_std()
        sd = self.c*currsd

        # Get the samples.
        if truncated:
            mn, mx = self.range
            Tnorm = get_truncated_normal(self.locs, sd, low=mn, upp=mx)
            proposals = Tnorm.rvs()
        else:
            proposals = np.random.normal(self.locs, scale = sd)
            
        return proposals
    
    def acceptance_rates(self, proposals, data):
        '''
        Calculate the Metropolis acceptance rates that assure detailed balance.
        '''
        # Calculate old and new loglikelihoods.
        olls = self.model.batch_likelihood(self.locs, data, self.var, 
                                           log = self.log, info="old")
        nlls = self.model.batch_likelihood(proposals, data, self.var, 
                                           log = self.log, info="new")
        if self.log:
            # Pass along proposals to print in case of weird behavior.
            acc_rates = self.llratios(nlls, olls, proposals)
        else:
            # keep: which proposals to keep (delete particle if both old and new 
            # likelihoods are 0)
            acc_rates, keep = self.ratios_catch(nlls, olls)
            proposals = proposals[keep]
        
        acc_rates = self.cap_acc_rates(acc_rates)
        # self.tune_rwm(acc_rates)

        return acc_rates, proposals#[keep]
    
    def tune_rwm(self, acc_rates):
        # Target interval for the acceptance rates. 
        tg = (0.2, 0.5)

        acc = np.mean(acc_rates)
        if acc < tg[0]:
            self.c *= 0.5
            print(f"> Decreased Metropolis factor to {self.c}.")
        if acc > tg[1]:
            self.c *= 1.1
            print(f"> Increased Metropolis factor to {self.c}.")
    
    def llratios(self, new, old, proposals):
        return self.exp_catch(new, old, proposals)
    
    def exp_catch(self, A, B, proposals):
        '''
        Do exp(A-B) and catch overflows.
        '''
        def cleanup(A, B, print_which = True):
            if print_which:
                for i, (a, b) in enumerate(zip(A,B)):
                    try:
                        np.exp(a-b)
                    except FloatingPointError as e:
                        print(f"> {e.args[0]} with {a = }, {b = } in exp_catch.", 
                              end = " ")
                        with np.errstate(all = "ignore"):
                            print(f"exp(a-b) = {np.exp(a-b)}")
                        print("> Particle, w with log-likelihood b: ", 
                              self.locs[i], f"(weight {self.ws[i]})")
                        print("> Proposal with log-likelihood a:    ", 
                              proposals[i])
                        input()
                    

        exp_calc = lambda A, B: np.exp(A-B)
        r = np_invalid_catch(f = exp_calc,
                                    largs = [A, B], 
                                    errdesc = ["overflow encountered in exp"], 
                                    cleanup = cleanup,
                                    caller = "exp_catch")
        return r
    
    def print_lists(self):
        print("> params, locs, rws, ws:")
        lprint(self.param_of_interest)
        lprint(self.locs)
        lprint(self.rws)
        lprint(self.ws)

    def ratios_catch(self, numerator, denominator, repby = 1):
        '''
        Divide numerator by denominator. If denominator is zero, replace
        division by 1 and accept proposal deterministically unless the 
        numerator is also 0 (in which case the particle is removed).
        
        Return numerator and denumerator because to pass on deletions.
        '''
        def cleanup(n, d):
            zero_pc = np.sum(d == 0)/len(d)
            print(f"> {100*zero_pc:.1f}% of old_likelihoods" 
                  + " values were zero. [sampler]")
            # If the denominator is zero, assume 1 acceptance probability. 
            if zero_pc > 0.9:
                self.print_lists()
            try:
                ratios = np.where(d==0, 1, n/d)
            except FloatingPointError:
                for x,y in zip(n,d):
                    try:
                        x/y
                    except FloatingPointError as e:
                        if y!=0:
                            print(f"> {e.args[0]} with {x = }, {y = } in ratios_catch.")
                            self.print_lists()
                        with np.errstate(all = "ignore"):
                            ratios = np.where(d==0, 1, n/d)
                    
            # If both denominator and numerator were zero, remove particles.
            keep = (n+d != 0)
            # Must do individually to actually modify array.
            self.locs, self.ws = self.locs[keep], self.ws[keep]
            ratios, n, d = ratios[keep], n[keep], d[keep]
            return ratios, keep
            
        # Probabilities.
        # numerator[0] = 0
        # denominator[0] = 0
        assert all(numerator>=0) and all(denominator>=0)
        # Returns a list of ratios + a mask for which ratios are valid.
        ratio_calc = lambda n, d: (np.where(d==0, repby, n/d), (n+d != 0))
        r, keep = np_invalid_catch(f = ratio_calc,
                                    largs = [numerator, denominator], 
                                    errdesc = ["divide by zero encountered in true_divide",
                                                "divide by zero encountered in divide",
                                                "invalid value encountered in true_divide",
                                                "invalid value encountered in divide"], 
                                    cleanup = cleanup,
                                    caller = "ratios_catch")
        return r, keep
    
    @staticmethod
    def cap_acc_rates(acc_rates):
        '''
        Caps > 1 values at 1:  Metropolis rate should be min(1, Pnew/Pold).
        '''
        if isinstance(acc_rates, tuple):
            print("tuple!", acc_rates)
        acc_rates = np.where(acc_rates>1, 1, acc_rates)
        return acc_rates

    def rejection_step(self, acc_rates, proposals):
        '''
        Probabilistically accept or reject the new samples.
        '''
        accept = np.random.binomial(1, acc_rates)
        self.locs = np.where(accept, proposals, self.locs)
        
        acc_rate = np.sum(accept)/len(proposals)
        self.acc_rates.append(acc_rate)
    
    def print_stats(self):
        super().print_stats()
        # Remove 'np.float64' part for printing.
        floatlist = [float(acc) for acc in self.acc_rates] 
        print(f"> Ordered acceptance rates ({len(floatlist)}): ", 
              floatlist)
        avg = np.sum(self.acc_rates)/len(self.acc_rates)
        print(f"> Average acceptance rate: {avg*100:.1f}%")
        # print(f"> Ordered ESS: ", self.ess_list)
        
def get_sampler(sampler_str, M, kwargs):
    '''
    Return a sampler of the indicated type. Can be LW, RWM, or a static grid.
    '''
    if sampler_str == "LW":
        sampler = LiuWestSampler(M, **kwargs) 
    elif sampler_str == "RWM":
        sampler = MetropolisSampler(M, **kwargs)
    elif sampler_str == "grid":
        sampler = LiuWestSampler(M, thr=0)
    return sampler