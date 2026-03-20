import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ractional_Brownian_Field_main import RandomFieldSimulator


# Sample Hurst-index pairs
# ============================================================
# Main purpose:
#   This block constructs representative pairs (H1, H2) for the two paths
#   B(t, H1) and B(t, H2) of the fractional Brownian field.
#
#   In this simulation study, the quantity H1 + H2 is the key diagnostic for
#   cross long-term memory. Hence, instead of choosing H1 and H2 arbitrarily,
#   we generate several regimes corresponding to:
#       (i)   H1 + H2 > 1
#       (ii)  H1 + H2 = 1
#       (iii) H1 + H2 < 1
#   together with two extreme benchmark cases where both Hurst indices are
#   simultaneously very small or very large.
#
# Interpretation:
#   - H close to 0  -> rougher path 
#   - H close to 1  -> smoother path
#   - H1 + H2 > 1   -> case associated with cross long-term memory
#   - H1 + H2 = 1   -> critical boundary case
#   - H1 + H2 < 1   -> case without cross long-term memory



def sample_H_pair_by_sum(S, rng, r=None, r_low=0.2, r_high=0.8, eps=0.05):
    """
    Generate a pair (H1, H2) such that H1 + H2 = S.

    Parameters:
        S : Prescribed target value of H1 + H2.
        rng : numpy.random.Generator
            Random number generator used for reproducible sampling.
    r : Fixes the splitting ratio so that H1 = r*S,  H2 = (1-r)*S.
        If None, the ratio is sampled uniformly from [r_low, r_high].
    r_low, r_high : float
        Lower and upper bounds for the random splitting ratio.
    eps : float
        Safety margin keeping both Hurst indices away from 0 and 1.

    Returns:
        A valid pair (H1, H2) satisfying H1 + H2 = S.

    Remarks:
        The range for a Hurst parameter is (0,1). The condition eps < H1 < 1-eps,   eps < H2 < 1-eps
        avoids degenerate choices too close to the boundary, where simulation 
        can become numerically unstable or less visually informative.
    """

    # Try repeatedly until a valid pair is obtained.
    for _ in range(10000):

        rr = r if r is not None else rng.uniform(r_low, r_high)

        H1 = rr * S
        H2 = (1-rr) * S

        if eps < H1 < 1-eps and eps < H2 < 1-eps:
            return float(H1), float(H2)

    raise RuntimeError("Cannot generate H pair")


def make_regimes(delta=0.2, seed=123):
    """
    Construct the list of Hurst parameter regimes used in the combined figure plots.

    Parameters:
        delta : float, optional
        seed : int, default=123
            Seed for reproducible random generation of the boundary cases.

    Returns:
        list of tuples -> (tag, H1, H2) where:
            - tag is a descriptive label for the regime,
            - H1 and H2 are the two Hurst indices.

    Included regimes:
        sum_gt_1     : a regime with H1 + H2 > 1
        sum_eq_1_a   : a first critical case with H1 + H2 = 1
        sum_eq_1_b   : a second critical case with H1 + H2 = 1
        sum_lt_1     : a regime with H1 + H2 < 1
        both_small_H : both indices small, yielding rough paths
        both_large_H : both indices large, yielding smooth paths
    """

    # Initialize the random generator for reproducibility.
    rng = np.random.default_rng(seed)

    # A representative regime with H1 + H2 > 1. Both Hurst parameters should deviate sufficiently from 0.5
    # to ensure a clear distinction from the critical cases
    H1_gt,H2_gt=0.85,0.35

    # Two critical boundary cases with H1 + H2 = 1.
    H1_eq_a,H2_eq_a = sample_H_pair_by_sum(1,rng,r=0.75)
    H1_eq_b,H2_eq_b = sample_H_pair_by_sum(1,rng,r=0.5)

    # A representative regime with H1 + H2 < 1.
    H1_lt,H2_lt=0.18,0.71

    regimes = [

        ("sum_gt_1",H1_gt,H2_gt),
        ("sum_eq_1_a",H1_eq_a,H2_eq_a),
        ("sum_eq_1_b",H1_eq_b,H2_eq_b),
        ("sum_lt_1",H1_lt,H2_lt),

        ("both_small_H",0.08,0.08),
        ("both_large_H",0.92,0.92)

    ]

    return regimes


# Simulate two paths
# ============================================================
# Main idea:
#   For a fixed time t, we simulate two processes: B(t, H1) and B(t, H2),
#   where both are viewed as slices of the same fractional Brownian field
#
# These two simulated paths are then used to build:
#   - the 3D trajectory (t, x1(t), x2(t)),
#   - the 2D projection (t, x1(t)),
#   - the 2D projection (t, x2(t)).

def simulate_two_slices(H1, H2,
                        sample_size=1000,
                        tmax=1,
                        FBM_cov_md=1,
                        rf_factor=0.7,
                        seed=123):
    """
    Simulate two trajectories B(t,H1) and B(t,H2) on the same time interval.

    Parameters:
        H1, H2 : The two Hurst parameters specifying the two slices of the field.
        sample_size : Number of grid points in time.
        tmax : Right endpoint of the time interval [0, tmax].
        FBM_cov_md : Covariance-model parameter passed to RandomFieldSimulator.
        rf_factor : Additional model parameter passed to RandomFieldSimulator.
        seed : Base random seed used for reproducibility.

    Returns:
        t : Uniform time grid on [0, tmax].
        x1 : Simulated sample path corresponding to H1.
        x2 : Simulated sample path corresponding to H2.
    """

    # Build two simulators, one for each Hurst index
    sim1 = RandomFieldSimulator(sample_size, H1, 0, tmax, FBM_cov_md, rf_factor)
    sim2 = RandomFieldSimulator(sample_size, H2, 0, tmax, FBM_cov_md, rf_factor)

    # Attempt reproducible simulation for the two paths.
    try:
        x1 = sim1.get_self_similar_process(seed=seed+1)
    except:
        x1 = sim1.get_self_similar_process()

    try:
        x2 = sim2.get_self_similar_process(seed=seed+2)
    except:
        x2 = sim2.get_self_similar_process()

    # Convert outputs to NumPy arrays to guarantee compatibility with plotting.
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)

    # Uniform time grid used for both trajectories.
    t = np.linspace(0, tmax, sample_size)

    return t, x1, x2


# Combined figure: 3D + 2 projections
# ============================================================
# Layout:
#   - Left panel      : 3D curve (t, B(t,H1), B(t,H2))
#   - Top-right panel : projection onto (t, B(t,H1))
#   - Bottom-right    : projection onto (t, B(t,H2))
#
# Purpose:
#   The 3D panel reveals the joint dynamics of the two paths, whereas the
#   projection panels isolate the regularity of each coordinate separately.
#   The joint geometry depends not only on H1 and H2 individually, but also on their sum.

def plot_combined_figure(t, x1, x2, H1, H2, title, outpath=None):
    """
    Plot a single combined figure containing one 3D trajectory and two 2D projections.

    Parameters:
        t : Time grid.
        x1, x2 : Simulated trajectories corresponding to B(t,H1) and B(t,H2).
        H1, H2 : Hurst parameters used to generate x1 and x2.
        title : Title for the full figure.
        outpath : If provided, the figure is saved to this path.
    """

    # Create the figure canvas.
    fig = plt.figure(figsize=(14, 8))

    # Define a 2x2 grid:
    #   - the left column spans both rows and hosts the 3D trajectory,
    #   - the right column contains the two 2D projections.
    gs = GridSpec(2, 2, width_ratios=[1.6, 1], height_ratios=[1, 1], figure=fig)


    # Left panel: 3D trajectory
    # This curve is the parametric trajectory: t ↦ (t, B(t,H1), B(t,H2)).
    # It simultaneously displays the interaction between the two slices and
    # therefore gives the clearest visual impression of the joint structure.
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.plot(t, x1, x2, linewidth=1, color="red")

    # Label the three coordinates of the trajectory.
    ax3d.set_xlabel(r"$t$", labelpad=10)
    ax3d.set_ylabel(r"$B(t,H_1)$", labelpad=10)
    ax3d.set_zlabel(r"$B(t,H_2)$", labelpad=10)

    # Include both the individual Hurst indices and their sum in the title
    ax3d.set_title(
        rf"3D trajectory $(t, B(t,H_1), B(t,H_2))$"
        "\n"
        rf"$H_1={H1:.2f},\; H_2={H2:.2f},\; H_1+H_2={H1+H2:.2f}$",
        pad=16
    )

    # Fix a viewing angle that gives a readable balance between depth 
    # and separation of the two vertical coordinates.
    ax3d.view_init(elev=25, azim=-60)


    # Top-right panel: projection onto (t, B(t,H1))
    # This is simply the first path viewed as a standard 2D trajectory.
    # It helps assess the path regularity associated with H1 alone.
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, x1, linewidth=1, color="red")
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$B(t,H_1)$")
    ax1.set_title(
        rf"Projection on $(t, B(t,H_1))$"
        "\n"
        rf"$H_1={H1:.2f}$"
    )


    # Bottom-right panel: projection onto (t, B(t,H2))
    # This panel plays the same role for the second path and allows a direct
    # visual comparison with the first projection.
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(t, x2, linewidth=1, color="red")
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$B(t,H_2)$")
    ax2.set_title(
        rf"Projection on $(t, B(t,H_2))$"
        "\n"
        rf"$H_2={H2:.2f}$"
    )

    fig.suptitle(title, fontsize=15, y=0.98)

    # Adjust subplot spacing to reduce overlap.
    plt.tight_layout()

    # Save the figure if an output path is provided.
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.2)

    plt.show()



# Main execution
# ============================================================================
# Workflow:
#   1. Create an output directory.
#   2. Generate the collection of Hurst-index cases.
#   3. Simulate two paths for each case.
#   4. Produce one combined figure per case.
#   5. Save all figures to disk.


def main():
    """
    Run the full simulation-and-plotting pipeline.
    """

    # Directory where all combined figures will be stored.
    outdir = "FBF_combined_plots"
    os.makedirs(outdir, exist_ok=True)

    # Build the list of test regimes.
    regimes = make_regimes()

    # Print the regimes to the console to verify the values used.
    print("Regimes used:")
    for tag, H1, H2 in regimes:
        print(tag, H1, H2, "sum =", H1+H2)

    # Loop over all regimes and generate one figure for each.
    for tag, H1, H2 in regimes:

        # Simulate the two slices B(t,H1) and B(t,H2).
        t, x1, x2 = simulate_two_slices(H1, H2)

        # File name for the saved plot corresponding to the current regime.
        outfile = os.path.join(outdir, f"{tag}_combined.png")

        # Produce and save the combined figure.
        plot_combined_figure(
            t, x1, x2,
            H1, H2,
            title="Fractional Brownian field trajectory and its projections",
            outpath=outfile
        )

    # Report the output directory location.
    print("\nCombined figures saved in:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()





