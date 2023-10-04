import numpy as np
import scipy.io
import math
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from itertools import permutations


def normalize_array(arr):
    min_vals = np.min(arr, axis=0)
    max_vals = np.max(arr, axis=0)
    normalized_arr = (arr - min_vals) / (max_vals - min_vals)
    return normalized_arr


class Local_Density_Peaks(object):

    def __init__(self, data):

        self.clusters = None
        self.r = None
        data = np.array(data)

        self.data = data
        self.data = normalize_array(self.data)
        self.n_id = data.shape[0]

        self.distances, self.Di, self.ds = self.nearest_neighbor_distance()
        self.density = self.local_density()
        self.ud = self.upward_distance()

        self.rnn = self.reverse_nearest_neighbors(10)

    def nearest_neighbor_distance(self):
        """
                Calculate distance between all points,
                distance each point and its nearest neighbor,
                maximum distance with nearest neighbor.

                :return: distance distances, Di, ds
        """
        distances = euclidean_distances(self.data)

        nearest_neighbor_distances = np.sort(distances, axis=1)[:, 1]

        return distances, nearest_neighbor_distances, max(nearest_neighbor_distances)

    def local_density(self):
        """
        Compute all points' local density.

        :return: local density vector that index is the point index
        """

        rho = [0] * self.n_id

        for i in range(self.n_id):
            for j in range(i + 1, self.n_id):
                if self.distances[i, j] <= self.ds:
                    temp = pow(math.e, (-1 * ((self.distances[i, j] / self.ds) ** 2)))
                    rho[i] += temp
                    rho[j] += temp

        return rho

    def upward_distance(self):
        """
                Calculate upward distance .

                :return: list
        """

        ud = [0] * self.n_id
        maxDensity = np.max(self.density)

        for begin in range(self.n_id):
            ud[begin] = np.inf
            if self.density[begin] < maxDensity:
                for end in range(self.n_id):
                    if self.density[end] > self.density[begin]:
                        ud[begin] = min(ud[begin], self.distances[begin][end])
            else:
                ud[begin] = np.max(self.distances[begin])

        return ud

    def reverse_nearest_neighbors(self, k):

        """
        :param k:
        :return list of RNN for each pont:
        """

        # Build the k-d tree
        tree = KDTree(self.data)

        # Calculate the k nearest neighbors for each point
        _, indices = tree.query(self.data, k=k + 1)  # +1 to include the point itself

        # Initialize an empty dictionary to store the reverse nearest neighbors
        reverse_neighbors = [0] * self.n_id

        # Iterate over each point
        for i in range(self.n_id):
            for j in indices[i]:
                if j != i:
                    reverse_neighbors[j] += 1

        return reverse_neighbors

    def initial_sub_cluster_construction(self):

        """

        :return: Points label C, the number of initial sub-clusters
        """

        # points label
        C = [-1] * self.n_id

        # number of initial sub-clusters
        ICC = 0
        mean_ud, std_ud = np.mean(self.ud), np.std(self.ud)
        mean_density, std_density = np.mean(self.density), np.std(self.density)
        mean_rnn, std_rnn = np.mean(self.rnn), np.std(self.rnn)

        # determine the noise points
        for i in range(self.n_id):
            if self.ud[i] > (mean_ud + std_ud) and self.density[i] < (
                    mean_density - std_density) \
                    and self.rnn[i] < (mean_rnn - std_rnn):
                C[i] = 0

        # determine the initial sub-cluster centers
        for i in range(self.n_id):
            if self.ud[i] > (mean_ud + std_ud) and C[i] != 0:
                ICC += 1
                C[i] = ICC

        # assign the remaining points to the initial sub-clusters
        clusters = {i: set() for i, e in enumerate(C) if e > 0}

        point_density = {i: self.density[i] for i, e in enumerate(C) if e == -1}
        sorted_density = sorted(point_density.items(), key=lambda x: x[1], reverse=True)

        for y1, p1 in enumerate(sorted_density):
            nearest_distance = np.inf
            nearest_cluster = None
            i, val = p1
            for j in clusters.keys():
                if self.density[j] > self.density[i] and self.distances[i, j] < nearest_distance:
                    nearest_distance = self.distances[i, j]
                    nearest_cluster = j
            for y2 in range(y1):
                j, val2 = sorted_density[y2]
                if self.distances[i, j] < nearest_distance:
                    nearest_distance = self.distances[i, j]
                    for c in clusters.keys():
                        if j in clusters[c]:
                            nearest_cluster = c
            clusters[nearest_cluster].add(i)

        # show clusters
        # centers = []
        # for cluster_center in clusters.keys():
        #     cluster_points = []
        #     centers.append(self.data[cluster_center])
        #     for c in clusters[cluster_center]:
        #         cluster_points.append(self.data[c])
        #
        #     cluster_points = np.array(cluster_points)
        #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1])
        # centers = np.array(centers)
        # plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='*', label='Centers')
        noise_points = [self.data[i] for i, e in enumerate(C) if e == 0]
        noise_points = np.array(noise_points)
        # if len(noise_points) > 0:
        #     plt.scatter(noise_points[:, 0], noise_points[:, 1], c='green', marker='x', label='Noises')
        # plt.title("initial sub-clusters")
        # plt.show()

        print(f'{len(noise_points)} noise points and {ICC} initial sub-cluster centers')

        self.clusters = clusters
        self.C = C
        self.ICC = ICC

        return C, ICC

    def sub_cluster_updating(self):
        """

        :return: Point labelC, the number of sub-clusters ICC
        """
        # number of false sub_cluster
        F_ICC = 0

        nd = [0] * self.ICC
        nc = [0] * self.ICC

        centers = np.array(list(self.clusters.keys()))

        for i, c in enumerate(centers):
            nc[i] = len(self.clusters[c])
            for j in range(self.n_id):
                if j != c and self.distances[c, j] < self.ds:
                    nd[i] += 1

        # delete the false sub-cluster centers from the initial sub-cluster centers
        points = set()
        for i, c in enumerate(centers):
            if nc[i] < (0.5 * nd[i]):
                self.ICC = self.ICC - 1
                F_ICC += 1
                self.C[c] = -1
                points.add(c)
                val = self.clusters.pop(c)
                points.update(val)

        # assigning the points in the false sub-clusters to their nearest neighboring sub-clusters.
        for p in points:
            nearest_distance = np.inf
            nearest_cluster = None
            for j in self.clusters.keys():
                if self.distances[p, j] < nearest_distance:
                    nearest_distance = self.distances[p, j]
                    nearest_cluster = j
                for n in self.clusters[j]:
                    if self.distances[p, n] < nearest_distance:
                        nearest_distance = self.distances[p, n]
                        nearest_cluster = j
            self.clusters[nearest_cluster].add(p)

        # show clusters
        centers = []
        # for cluster_center in self.clusters.keys():
        #     cluster_points = []
        #     centers.append(self.data[cluster_center])
        #     for c in self.clusters[cluster_center]:
        #         cluster_points.append(self.data[c])
        #
        #     cluster_points = np.array(cluster_points)
        #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1])
        # centers = np.array(centers)
        # plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='*', label='Centers')
        noise_points = [self.data[i] for i, e in enumerate(self.C) if e == 0]
        noise_points = np.array(noise_points)
        # if len(noise_points) > 0:
        #     plt.scatter(noise_points[:, 0], noise_points[:, 1], c='green', marker='x', label='Noises')
        # plt.title("updating sub-clusters")
        # plt.show()

        print(f'{self.ICC} true sub-cluster centers indicated as blue stars with {F_ICC} false sub-cluster')

        return self.C, self.ICC

    def sub_cluster_merging(self):
        """

        :return: final clusters
        """

        self.r = self.radius()

        centers = np.array(list(self.clusters.keys()))
        merge_list = [[i] for i in self.clusters.keys()]
        for m in range(self.ICC - 1):
            for n in range(m + 1, self.ICC):
                c1 = centers[m]
                c2 = centers[n]
                d, xi, xj = self.cluster_distance(c1, c2)
                if d > self.r:
                    continue
                else:
                    if xi not in self.boundary_points(c1) and xj not in self.boundary_points(c2):
                        merge_list.append([c1, c2])
                    else:
                        if self.density[xi] + self.density[xj] > ((self.density[c1] + self.density[c2]) / 2):
                            merge_list.append([c1, c2])
                        else:
                            continue

        G = nx.Graph()
        for m in merge_list:
            nx.add_path(G, m)
        merge_list = list(nx.connected_components(G))

        # merge the corresponding clusters into one
        final_cluster = []
        for m in merge_list:
            sub = set()
            for i in m:
                sub.add(i)
                sub.update(self.clusters[i])
            final_cluster.append(sub)

        # Assign the noise points to the clusters
        noise_points = [i for i, e in enumerate(self.C) if e == 0]
        for n in noise_points:
            min_dis = np.inf
            num_cluster = None
            for i, c in enumerate(final_cluster):
                for p in c:
                    if self.distances[n, p] < min_dis:
                        min_dis = self.distances[n, p]
                        num_cluster = i
            final_cluster[num_cluster].add(n)

        # show clusters
        # for c in final_cluster:
        #     cluster_points = []
        #     for i in c:
        #         cluster_points.append(self.data[i])
        #
        #     cluster_points = np.array(cluster_points)
        #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1])
        # plt.title("Final result")
        # plt.show()

        return final_cluster

    def radius(self):

        if 0 not in self.C:
            r = self.ds
        else:
            r = 0
            for i, c in enumerate(self.C):
                if c != 0 and self.Di[i] > r:
                    r = self.Di[i]
        return r

    def cluster_distance(self, center1, center2):

        """

        :param center1:
        :param center2:
        :return: distance between cluster1 and cluster2
        """

        distance = np.inf
        x1 = None
        x2 = None
        for point1 in self.clusters[center1]:
            for point2 in self.clusters[center2]:
                if distance >= self.distances[point1, point2]:
                    distance = self.distances[point1, point2]
                    x1, x2 = point1, point2
        return distance, x1, x2

    def boundary_points(self, cluster):
        """

        :param cluster:
        :return boundary points in a cluster:
        """
        # calculate average density in cluster
        sum_density = 0
        num = 0
        for i in self.clusters[cluster]:
            sum_density += self.density[i]
            num += 1
        avg_density = sum_density / num

        boundary_points = []
        for i in self.clusters[cluster]:
            if self.density[i] < avg_density:
                boundary_points.append(i)

        return boundary_points


def evaluation(clusters, true_labels):
    permut = list(permutations(range(1, len(clusters) + 1)))

    # Assign labels to data points based on the cluster they belong to
    max_acc = 0
    best_label = None
    for p in permut:
        labels = [0] * len(true_labels)

        for i, cluster in enumerate(clusters):
            for j in cluster:
                labels[j] = p[i]
        accuracy = accuracy_score(true_labels, labels)
        if accuracy > max_acc:
            max_acc = accuracy
            best_label = labels

    accuracy = accuracy_score(true_labels, best_label)
    print("Accuracy:", accuracy)

    # Recall
    recall = recall_score(true_labels, best_label, average='macro')
    print("Recall:", recall)

    # Normalized Mutual Information
    nmi = normalized_mutual_info_score(true_labels, best_label)
    print("Normalized Mutual Information:", nmi)

    # Number of Clusters
    num_clusters = len(set(best_label))
    print("Number of Clusters:", num_clusters)



#mat = scipy.io.loadmat('../Data/gaussian.mat')
#mat = scipy.io.loadmat('../Data/ids2.mat')

# con_list = [[element for element in upperElement] for upperElement in mat['data']]
#
# columns = ['data_x', 'data_y']
# df = pd.DataFrame(con_list, columns=columns)


df = pd.read_csv("../Data/new-thyroid.data", sep=",", header=None)



ldp = Local_Density_Peaks(df.iloc[:, [i for i in range(1, 6)]])
# ldp = Local_Density_Peaks(df.iloc[:, [0,1]])
C, ICC = ldp.initial_sub_cluster_construction()
C, ICC = ldp.sub_cluster_updating()

final_cluster = ldp.sub_cluster_merging()

# labels = mat['label']
# labels = labels.flatten()
labels = df[0].to_numpy()
evaluation(final_cluster, labels)

