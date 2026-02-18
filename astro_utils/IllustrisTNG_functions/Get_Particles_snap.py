import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from astropy.table import vstack, Table
import glob
import os
import illustris_python as il

############# Extract Tables ################

def extract_fuzz_particles_with_subindex(
    base_sim,
    base_out,
    snap_number,
    center,
    vcm,
    radius,
    box_size,
    halo_id_arr,
    idsub,                   # [[id0_main, id1_main]]
    part_types=[4],
    part_names=['star'],
    loops=15
):
    """
    Extract particles within a sphere around 'center' that do NOT belong
    to massive subhaloes, using chunked reading for memory efficiency.
    Adds a 'subindex' column tagging each particle to one of the main
    subhaloes or 0 if it belongs to none.
    """

    print(f"\nExtracting {', '.join(part_names)} particles within {radius/1000:.2f} Mpc/h sphere.")

    snap_file = f"{base_sim}/simulation.hdf5"

    # --- Read header info (NumPart_Total, etc.) ---
    with h5py.File(snapPath(base_sim + 'output', 99), 'r') as f:
        header = dict(f['Header'].attrs.items())
        nPart = getNumPart(header)


    with h5py.File(snap_file, 'r') as f:
        
        # --- Main extraction loop ---
        for m, part_type in enumerate(part_types):
                part_name = part_names[m]
                print(f"\n--- Processing particle type: {part_name} (Type {part_type}) ---")
            
                # --- Vectorised exclusion indices ---
                print("Building exclusion list (particles in massive subhaloes)...")
        
                #part_type = part_types[0]
                offsets = f[f'/Offsets/{snap_number}/Subhalo/SnapByType'][:, part_type]
                lengths = f[f'/Groups/{snap_number}/Subhalo/SubhaloLenType'][:, part_type]
        
                starts = offsets[halo_id_arr]
                lens   = lengths[halo_id_arr]
                exclude_idx = np.concatenate([np.arange(s, s + l, dtype=np.int64) for s, l in zip(starts, lens)])
                exclude_idx = np.unique(exclude_idx)
                print(f"  → Total excluded: {len(exclude_idx):,} particles")
        
                print(f"Total particles by type: {nPart}")
        
                # --- Build a fast lookup table for subhalo membership ---
                total_particles = nPart[part_type]
                subindex_lookup = np.zeros(total_particles, dtype=np.int16)  # 0 = none
        
                # Fill subindex for two main subhaloes efficiently
                for sid, subid in enumerate(idsub):  # expects shape [[id0, id1]]
                    start = offsets[subid]
                    length = lengths[subid]
                    subindex_lookup[start:start+length] = sid + 1  # 1 or 2
                print("  → Subindex lookup table built.")
    
    
                total_n = nPart[part_type]
                bin_size = int(total_n / loops)
    
                coords = f[f'/Snapshots/{snap_number}/PartType{part_type}/Coordinates']
                vels   = f[f'/Snapshots/{snap_number}/PartType{part_type}/Velocities']
    
                print(f"\nProcessing {part_name} ({total_n:,} particles)...")
                conta = 0
    
                for i in tqdm(range(loops), desc=f"{part_name} chunks"):
                    start = i * bin_size
                    stop  = total_n if (i == loops - 1) else (i + 1) * bin_size
                    idx_range = np.arange(start, stop, dtype=np.int64)
    
                    # Exclude massive subhalo particles
                    mask_keep = np.isin(idx_range, exclude_idx, invert=True)
                    idx_valid = idx_range[mask_keep]
                    if len(idx_valid) == 0:
                        continue
    
                    pos = coords[idx_valid, :]
                    vel = vels[idx_valid, :]
    
                    # Periodic wrapping
                    delta = pos - center
                    np.abs(delta, out=delta)
                    np.minimum(delta, box_size - delta, out=delta)
                    np.square(delta, out=delta)
                    dist = np.sqrt(np.sum(delta, axis=1))
    
                    w = dist < radius
                    if not np.any(w):
                        continue
    
                    pos_sel = pos[w, :]
                    vel_sel = vel[w, :]
                    subindex_sel = subindex_lookup[idx_valid[w]]
    
                    # --- Save to FITS ---
                    fname_pos = f"{base_out}/Radial_Fuzz_positions_{part_name}_snap{snap_number}_{conta}.fits"
                    fname_vel = f"{base_out}/Radial_Fuzz_velocity_{part_name}_snap{snap_number}_{conta}.fits"
    
                    tpos = Table(
                        [pos_sel[:, 0], pos_sel[:, 1], pos_sel[:, 2], subindex_sel],
                        names=('px', 'py', 'pz', 'subindex')
                    )
                    tvel = Table(
                        [vel_sel[:, 0], vel_sel[:, 1], vel_sel[:, 2]],
                        names=('vx', 'vy', 'vz')
                    )
    
                    tpos.write(fname_pos, format='fits', overwrite=True)
                    tvel.write(fname_vel, format='fits', overwrite=True)
    
                    conta += 1
    
                    del pos, vel, delta, dist, w, pos_sel, vel_sel, subindex_sel
    
        print("\n✅ Extraction complete with subindex tagging.")


########## Merge tables ###########



def merge_fuzz_tables(base_out, part_name, snap_number):
    """
    Merge all chunked position and velocity FITS files for a given particle type,
    preserving 'subindex' column if present.

    Parameters
    ----------
    base_out : str
        Output directory containing the chunked FITS files.
    part_name : str
        Particle type name (e.g., 'star', 'gas', 'dm').
    snap_number : int
        Snapshot number to merge (e.g., 99).
    """

    # === Find all chunked files ===
    pos_files = sorted(glob.glob(f"{base_out}/Radial2_Fuzz_positions_{part_name}_snap{snap_number}_*.fits"))
    vel_files = sorted(glob.glob(f"{base_out}/Radial2_Fuzz_velocity_{part_name}_snap{snap_number}_*.fits"))

    print(f"Found {len(pos_files)} position files and {len(vel_files)} velocity files.")

    if len(pos_files) == 0 or len(vel_files) == 0:
        print("No matching files found — check filenames or paths.")
        return

    # === Read and stack ===
    pos_tables = []
    vel_tables = []

    for fpos, fvel in zip(pos_files, vel_files):
        tpos = Table.read(fpos)
        tvel = Table.read(fvel)
        pos_tables.append(tpos)
        vel_tables.append(tvel)

    print("Stacking tables... (this might take a bit for large files)")
    pos_all = vstack(pos_tables, metadata_conflicts='silent')
    vel_all = vstack(vel_tables, metadata_conflicts='silent')

    # === Merge into a single table ===
    merged_table = pos_all
    merged_table['vx'] = vel_all['vx']
    merged_table['vy'] = vel_all['vy']
    merged_table['vz'] = vel_all['vz']

    # === If subindex exists, keep it ===
    if 'subindex' in pos_all.colnames:
        print("Detected 'subindex' column — preserving it in merged table.")
    else:
        print("⚠️ No 'subindex' column found in position tables.")

    # === Save final merged file ===
    out_file = f"{base_out}/Radial2_Fuzz_{part_name}_snap{snap_number}_merged.fits"
    merged_table.write(out_file, format='fits', overwrite=True)

    print(f"✅ Merged table saved to: {out_file}")
    print(f"   Total particles: {len(merged_table):,}")
    if 'subindex' in merged_table.colnames:
        unique_subs = len(np.unique(merged_table['subindex']))
        print(f"   Unique subindex values: {unique_subs}")

############ Plot Tables Merger #############
def plot_merged_particles(fits_file, idsub, i, s=1e-3, subsample=None, figsize=(8,8)):
    """
    Plot the merged particle table in a 3D scatter projection,
    colouring points based on subindex.

    Parameters
    ----------
    fits_file : str
        Path to merged FITS file.
    idsub : array-like
        Array of subhalo ID pairs, e.g. [[1234, 5678], [9012, 3456], ...].
    i : int
        Index to select which idsub[i] pair to highlight.
    s : float
        Marker size for scatter plot.
    subsample : int, optional
        Number of random particles to plot (for speed).
    figsize : tuple
        Figure size in inches.
    """
    print(f"Loading data from: {fits_file}")
    t = Table.read(fits_file)

    # === Extract coordinates ===
    x, y, z = t['px'], t['py'], t['pz']
    subindex = t['subindex'] if 'subindex' in t.colnames else np.zeros(len(x), dtype=int)

    # === Optional subsampling ===
    if subsample and len(x) > subsample:
        idx = np.random.choice(len(x), subsample, replace=False)
        x, y, z, subindex = x[idx], y[idx], z[idx], subindex[idx]
        print(f"Subsampled to {subsample:,} points")

    # === Define masks for colouring ===
    sub0, sub1 = idsub[i][0], idsub[i][1]
    mask0 = subindex == sub0
    mask1 = subindex == sub1
    mask_other = ~(mask0 | mask1)
    print(len(mask1))
    # === Create figure ===
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Grey background particles
    ax.scatter(x[mask_other], y[mask_other], z[mask_other], s=s, c='0.6', alpha=0.3)
    # Red and blue for selected subhaloes
    ax.scatter(x[mask0], y[mask0], z[mask0], s=s, c='r', alpha=0.8, label=f"Subhalo {sub0}")
    ax.scatter(x[mask1], y[mask1], z[mask1], s=s, c='b', alpha=0.8, label=f"Subhalo {sub1}")

    ax.set_xlabel('x [Mpc/h]')
    ax.set_ylabel('y [Mpc/h]')
    ax.set_zlabel('z [Mpc/h]')
    ax.set_title(f'3D Fuzz Particles (pair {i}: {sub0}, {sub1})')

    # Equal aspect ratio
    max_range = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min())
    mid_x, mid_y, mid_z = x.mean(), y.mean(), z.mean()
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()


def snap_phy_subhalo(base, snap, subhalo_id, ptype, aexp, adot, hh, extra_fields=None):
    """Loads a particle type from TNG and converts to physical units."""
    fields = ['Velocities', 'Coordinates']
    if extra_fields:
        fields += extra_fields
    part = il.snapshot.loadSubhalo(base, snap, subhalo_id, ptype, fields=fields)

    # Convert positions and velocities to physical units
    pos = part['Coordinates'] / hh * aexp
    vel = part['Velocities'] * aexp**0.5 + pos * adot / hh

    data = {'pos': pos, 'vel': vel}
    if 'Masses' in part:
        data['mass'] = part['Masses'] * 1e10 / hh
    if 'Potential' in part:
        data['pot'] = part['Potential'] / aexp
    if 'GFM_Metallicity' in part:
        data['GFM_Metallicity'] = part['GFM_Metallicity']
    if 'InternalEnergy' in part:
        data['InternalEnergy'] = part['InternalEnergy']
    if 'ElectronAbundance' in part:
        data['ElectronAbundance'] = part['ElectronAbundance']
    if 'Density' in part:
        data['Density'] = part['Density'] * ((hh**2) *1e10) / (aexp**3.0)
    if 'StarFormationRate' in part:
        data['StarFormationRate'] = part['StarFormationRate']
    return data
    
