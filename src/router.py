import numpy as np
from sklearn.cluster import AgglomerativeClustering

class MachineTopology:
    """
    Physical Address Table for the Dual-Anchor Engine.
    Maps local branch columns back to global sensor indices.
    """
    def __init__(self, idx_phy, idx_res, idx_lone, idx_dead):
        self.idx_phy = np.array(idx_phy)
        self.idx_res = np.array(idx_res)
        self.idx_lone = np.array(idx_lone)
        self.idx_dead = np.array(idx_dead)
        self.res_to_dead_local = [
            np.where(self.idx_res == gid)[0][0] for gid in self.idx_dead
        ]
        self.res_to_lone_local = [
            np.where(self.idx_res == gid)[0][0] for gid in self.idx_lone
        ]

    def summary(self):
        print("-" * 35)
        print(f"MACHINE TOPOLOGY")
        print("-" * 35)
        print(f"Correlated Branch (HNN) : {len(self.idx_phy)} features")
        print(f"Residual Branch (FNO)  : {len(self.idx_res)} total")
        print("-" * 35)

def route_features(train_data, test_data, dist_threshold=0.5):

    num_total_features = train_data.shape[1]
    all_indices = np.arange(num_total_features)

    #1. Dead Sensor Extraction 
    variances = np.var(train_data, axis=0)
    idx_dead = np.where(variances < 1e-9)[0]
    idx_active = np.setdiff1d(all_indices, idx_dead)

    # If fewer than two sensors remain active after dead-sensor filtering
    # (i.e. len(idx_active) is 0 or 1), correlation clustering is undefined,
    # so every feature is routed as an isolated residual feature in the FNO branch.
    if len(idx_active) < 2:
        print(f" All dead sensors detected: {len(idx_active)} active features.")
        idx_phy = np.array([], dtype=int)
        idx_lone = all_indices
        idx_res = all_indices
        idx_dead = np.array([], dtype=int)
        phy_cluster_labels = np.array([], dtype=int)

        topo = MachineTopology(idx_phy, idx_res, idx_lone, idx_dead)
        topo.summary()
        print(f"phy_indices: {idx_phy}")
        
        return (train_data[:, :0], train_data, test_data[:, :0], test_data), topo, phy_cluster_labels

    # 2. Sensor Discovery 
    train_active = train_data[:, idx_active]
    corr = np.nan_to_num(np.corrcoef(train_active, rowvar=False))
    dist = 1 - np.abs(corr)
    if dist.ndim == 0:
        dist = dist.reshape(1, 1)
    
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=dist_threshold, 
        linkage='complete',
        metric="precomputed"
    )
    labels = clustering.fit_predict(dist)

    #3. Cluster Filtering 
    u_ids, counts = np.unique(labels, return_counts=True)
    consensus_ids = u_ids[counts > 1]
    is_consensus = np.isin(labels, consensus_ids)
    phy_cluster_labels = labels[is_consensus]

    # 4. Final Routing & Topology
    idx_phy = idx_active[is_consensus]
    idx_lone = np.sort(np.concatenate([idx_active[~is_consensus], idx_dead]))
    idx_res = idx_lone
    idx_dead = np.array([], dtype=int)

    topo = MachineTopology(idx_phy, idx_res, idx_lone, idx_dead)
    topo.summary()
    print(f"phy_indices: {idx_phy}")

    # 5. Data Splitting
    train_phy, train_res = train_data[:, idx_phy], train_data[:, idx_res]
    test_phy, test_res = test_data[:, idx_phy], test_data[:, idx_res]

    return (train_phy, train_res, test_phy, test_res), topo, phy_cluster_labels
