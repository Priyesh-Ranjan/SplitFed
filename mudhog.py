import numpy as np
from sklearn.cluster import DBSCAN, KMeans
#import time
from collections import defaultdict, Counter
import torch
from codecarbon import EmissionsTracker
import logging
from copy import deepcopy
from sklearn.decomposition import PCA

class MuDHoG():
    def __init__(self) :
        self.iter = 0
        self.tao_0 = 0
        self.dbscan_eps = 0.5
        self.mal_ids = set()
        self.dbscan_min_samples = 5
        self.flip_sign_ids = set()
        #self.uAtk_ids = set()
        self.delay_decision = 2
        self.sims = None
        #self.mal_ids = set()
        self.uAtk_ids = set()
        self.tAtk_ids = set()
        self.flip_sign_ids = set()
        #self.unreliable_ids = set()
        self.suspicious_id = set()
        self.unreliable_ids = set()
        self.pre_mal_id = defaultdict(int)
        self.count_unreliable = defaultdict(int)
    
    def aggregator(self, w, objects):
        tracker = EmissionsTracker(log_level=logging.CRITICAL)
        tracker.start()
        if isinstance(objects,list) :
            clients = objects
            for client in clients:
                client.compute_hogs()
            weight_vals = self.calculator(*self.get_hogs(clients, 'clients'))    
            print("Client-side aggregation done")
        else :
            server = objects
            server.compute_hogs()
            weight_vals = self.calculator(*self.get_hogs(server, 'server'))
            print("Server-side aggregation done")
        w_avg = deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] *= 0
            for i in range(0, len(w)):
                w_avg[k] += w[i][k] * weight_vals[i]
            w_avg[k] = torch.div(w_avg[k], np.sum(weight_vals))
        agg: float = tracker.stop()
        return w_avg, agg    
    
    def find_majority_id(self, clf):
        counts = Counter(clf.labels_)
        major_label = max(counts, key=counts.get)
        major_id = np.where(clf.labels_ == major_label)[0]
        #major_id = set(major_id.reshape(-1))
        return major_id
    
    def find_minority_id(self, clf):
        count_1 = sum(clf.labels_ == 1)
        count_0 = sum(clf.labels_ == 0)
        mal_label = 0 if count_1 > count_0 else 1
        atk_id = np.where(clf.labels_ == mal_label)[0]
        atk_id = set(atk_id.reshape((-1)))
        return atk_id
    
    def find_separate_point(self, d):
        # d should be flatten and np or list
        d = sorted(d)
        sep_point = 0
        max_gap = 0
        for i in range(len(d)-1):
            if d[i+1] - d[i] > max_gap:
                max_gap = d[i+1] - d[i]
                sep_point = d[i] + max_gap/2
        return sep_point
    
    def find_targeted_attack(self, dict_lHoGs):
        """Construct a set of suspecious of targeted and unreliable client's
        by using long HoGs (dict_lHoGs dictionary).
        - cluster: Using KMeans (K=2) based on Euclidean distance of
        long_HoGs==> find minority ids.
        """
        id_lHoGs = np.array(list(dict_lHoGs.keys()))
        value_lHoGs = np.array(list(dict_lHoGs.values()))
        cluster_lh1 = KMeans(n_clusters=2, random_state=0).fit(value_lHoGs)
        offset_tAtk_id1 = self.find_minority_id(cluster_lh1)
        sus_tAtk_id = id_lHoGs[list(offset_tAtk_id1)]
        return sus_tAtk_id

#=======================================================================================================

    def get_hogs(self, entity, nature) :
        # long_HoGs for clustering targeted/untargeted and calculating angle > 90 for flip-sign attack
        long_HoGs = {}

        # normalized_sHoGs for calculating angle > 90 for flip-sign attack
        #full_norm_short_HoGs = [] # for scan flip-sign each round

        # L2 norm short HoGs for detecting additive noise or Gaussian/random noise untargeted
        short_HoGs = {}
        
        normalized_sHoGs = {}

        if nature == "clients" : self.num_clients = len(entity)
        elif nature == "server" : self.num_clients = entity.get_num_clients()
        
        # STAGE 1: Collect long and short HoGs.
        for i in range(self.num_clients):
            # longHoGs
            if nature == 'clients' :
                sum_hog_i = entity[i].get_sum_hog().detach().cpu().numpy()
            elif nature == 'server' :
                sum_hog_i = entity.get_sum_hog(i).detach().cpu().numpy()
            #L2_sum_hog_i = client's[i].get_L2_sum_hog().detach().cpu().numpy()
            long_HoGs[i] = sum_hog_i

            # shortHoGs
            if nature == 'clients' :
                sHoG = entity[i].get_avg_grad().detach().cpu().numpy()
            elif nature == 'server' :
                sHoG = entity.get_avg_grad(i).detach().cpu().numpy()
            #sHoG = clients[i].get_avg_grad().detach().cpu().numpy()
            L2_sHoG = np.linalg.norm(sHoG)
            #full_norm_short_HoGs.append(sHoG/L2_sHoG)
            short_HoGs[i] = sHoG
            
            if i not in self.mal_ids:
                normalized_sHoGs[i] = sHoG/L2_sHoG
                
        return short_HoGs, long_HoGs, normalized_sHoGs

    def calculator(self, short_HoGs, long_HoGs, normalized_sHoGs):
        
        #short_HoGs, long_HoGs, normalized_sHoGs = self.get_hogs(clients)
        
        # Exclude the firmed malicious client's
                
        #self.num_clients = len(clients)
        
        # STAGE 2: Clustering and find malicious client's
        if self.iter >= self.tao_0:
            # STEP 1: Detect FLIP_SIGN gradient attackers
            """By using angle between normalized short HoGs to the median
            of normalized short HoGs among good candidates.
            NOTE: we tested finding flip-sign attack with longHoG, but it failed after long running.
            """
            flip_sign_id = set()
            """
            median_norm_shortHoG = np.median(np.array([v for v in normalized_sHoGs.values()]), axis=0)
            for i, v in enumerate(full_norm_short_HoGs):
                dot_prod = np.dot(median_norm_shortHoG, v)
                if dot_prod < 0: # angle > 90
                    flip_sign_id.add(i)
                    #logging.debug("Detect FLIP_SIGN client={}".format(i))
            logging.info(f"flip_sign_id={flip_sign_id}")
            """
            non_mal_sHoGs = dict(short_HoGs) # deep copy dict
            for i in self.mal_ids:
                non_mal_sHoGs.pop(i)
            median_sHoG = np.median(np.array(list(non_mal_sHoGs.values())), axis=0)
            for i, v in short_HoGs.items():
                v = np.array(list(v))
                d_cos = np.dot(median_sHoG, v)/(np.linalg.norm(median_sHoG)*np.linalg.norm(v))
                if d_cos < 0: # angle > 90
                    flip_sign_id.add(i)

#=======================================================================================================

            # STEP 2: Detect UNTARGETED ATTACK
            """ Exclude sign-flipping first, the remaining nodes include
            {NORMAL, ADDITIVE-NOISE, TARGETED and UNRELIABLE}
            we use DBSCAN to cluster them on raw gradients (raw short HoGs),
            the largest cluster is normal client's cluster (C_norm). For the remaining raw gradients,
            compute their Euclidean distance to the centroid (mean or median) of C_norm.
            Then find the bi-partition of these distances, the group of smaller distances correspond to
            unreliable, the other group correspond to additive-noise (Assumption: Additive-noise is fairly
            large (since it is attack) while unreliable's noise is fairly small).
            """

            # Step 2.1: excluding sign-flipping nodes from raw short HoGs:
            for i in range(self.num_clients):
                if i in flip_sign_id or i in self.flip_sign_ids:
                    short_HoGs.pop(i)
            id_sHoGs, value_sHoGs = np.array(list(short_HoGs.keys())), np.array(list(short_HoGs.values()))
            # Find eps for MNIST and CIFAR:
            #print(np.shape(value_sHoGs))
            # DBSCAN is mandatory success for this step, KMeans failed.
            # MNIST uses default eps=0.5, min_sample=5
            # CIFAR uses eps=50, min_sample=5 (based on heuristic evaluation Euclidean distance of grad of RestNet18.
            #start_t = time.time()
            pca = PCA(n_components=self.num_clients)
            value_sHoGs = pca.fit_transform(value_sHoGs)
            cluster_sh = DBSCAN(eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples).fit(value_sHoGs)
            #t_dbscan = time.time() - start_t
            
            offset_normal_ids = self.find_majority_id(cluster_sh)
            normal_ids = id_sHoGs[list(offset_normal_ids)]
            normal_sHoGs = value_sHoGs[list(offset_normal_ids)]
            normal_cent = np.median(normal_sHoGs, axis=0)


            # suspicious ids of untargeted attacks and unreliable or targeted attacks.
            offset_uAtk_ids = np.where(cluster_sh.labels_ == -1)[0]
            sus_uAtk_ids = id_sHoGs[list(offset_uAtk_ids)]


            # suspicious_ids consists both additive-noise, targeted and unreliable client's:
            suspicious_ids = [i for i in id_sHoGs if i not in normal_ids] # this includes sus_uAtk_ids

            d_normal_sus = {} # distance from centroid of normal to suspicious client's.
            for sid in suspicious_ids:
                d_normal_sus[sid] = np.linalg.norm(short_HoGs[sid]-normal_cent)

            # could not find separate points only based on suspected untargeted attacks.
            
            d_separate = self.find_separate_point(list(d_normal_sus.values()))
            sus_tAtk_uRel_id0, uAtk_id = set(), set()
            for k, v in d_normal_sus.items():
                if v > d_separate and k in sus_uAtk_ids:
                    uAtk_id.add(k)
                else:
                    sus_tAtk_uRel_id0.add(k)

#=======================================================================================================

            # STEP 3: Detect TARGETED ATTACK
            """
              - First excluding flip_sign and untargeted attack from.
              - Using KMeans (K=2) based on Euclidean distance of
                long_HoGs==> find minority ids.
            """
            for i in range(self.num_clients):
                if i in self.flip_sign_ids or i in flip_sign_id:
                    if i in long_HoGs:
                        long_HoGs.pop(i)
                if i in uAtk_id or i in self.uAtk_ids:
                    if i in long_HoGs:
                        long_HoGs.pop(i)

            # Using Euclidean distance is as good as cosine distance (which used in MNIST).
            tAtk_id = self.find_targeted_attack(long_HoGs)

            # Aggregate, count and record ATTACKERs:
            self.add_mal_id(flip_sign_id, uAtk_id, tAtk_id)

#=======================================================================================================

            # STEP 4: UNRELIABLE CLIENTS
            """using normalized short HoGs (normalized_sHoGs) to detect unreliable client's
            1st: remove all malicious client's (manipulate directly).
            2nd: find angles between normalized_sHoGs to the median point
            which mostly normal point and represent for aggreation (e.g., Median method).
            3rd: find confident mid-point. Unreliable client's have larger angles
            or smaller cosine similarities.
            """
            
            for i in self.mal_ids:
                if i in short_HoGs:
                    short_HoGs.pop(i)

            angle_sHoGs = {}
            # update this value again after excluding malicious client's
            median_sHoG = np.median(np.array(list(short_HoGs.values())), axis=0)
            for i, v in short_HoGs.items():
                angle_sHoGs[i] = np.dot(median_sHoG, v)/(np.linalg.norm(median_sHoG)*np.linalg.norm(v))

            angle_sep_sH = self.find_separate_point(list(angle_sHoGs.values()))
            normal_id, uRel_id = set(), set()
            for k, v in angle_sHoGs.items():
                if v < angle_sep_sH: # larger angle, smaller cosine similarity
                    uRel_id.add(k)
                else:
                    normal_id.add(k)

            for k in range(self.num_clients):
                if k in uRel_id:
                    self.count_unreliable[k] += 1
                    if self.count_unreliable[k] > self.delay_decision:
                        self.unreliable_ids.add(k)
                # do this before decreasing count
                if self.count_unreliable[k] == 0 and k in self.unreliable_ids:
                    self.unreliable_ids.remove(k)
                if k not in uRel_id and self.count_unreliable[k] > 0:
                    self.count_unreliable[k] -= 1

            weight_vec = np.ones(self.num_clients)
            for i in range(self.num_clients):
                if i not in self.mal_ids and i not in tAtk_id and i not in uAtk_id:
                    weight_vec[i] = 1
                else : weight_vec[i] = 0    
            #self.normal_clients = normal_clients
        else:
            weight_vec = np.ones(self.num_clients)
        
        self.iter += 1    
        print(weight_vec)
        #out = self.FedFuncWholeNet(normal_clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return weight_vec
    
    def add_mal_id(self, sus_flip_sign, sus_uAtk, sus_tAtk):
        all_suspicious = sus_flip_sign.union(sus_uAtk, sus_tAtk)
        for i in range(self.num_clients):
            if i not in all_suspicious:
                if self.pre_mal_id[i] == 0:
                    if i in self.mal_ids:
                        self.mal_ids.remove(i)
                    if i in self.flip_sign_ids:
                        self.flip_sign_ids.remove(i)
                    if i in self.uAtk_ids:
                        self.uAtk_ids.remove(i)
                    if i in self.tAtk_ids:
                        self.tAtk_ids.remove(i)
                else: #> 0
                    self.pre_mal_id[i] = 0
                    # Unreliable clients:
                    if i in self.uAtk_ids:
                        self.count_unreliable[i] += 1
                        if self.count_unreliable[i] >= self.delay_decision:
                            self.uAtk_ids.remove(i)
                            self.mal_ids.remove(i)
                            self.unreliable_ids.add(i)
            else:
                self.pre_mal_id[i] += 1
                if self.pre_mal_id[i] >= self.delay_decision:
                    if i in sus_flip_sign:
                        self.flip_sign_ids.add(i)
                        self.mal_ids.add(i)
                    if i in sus_uAtk:
                        self.uAtk_ids.add(i)
                        self.mal_ids.add(i)
                if self.pre_mal_id[i] >= 2*self.delay_decision and i in sus_tAtk:
                    self.tAtk_ids.add(i)
                    self.mal_ids.add(i)