import scipy
import pandas as pd


def compute_distance_matrix_internal(df: pd.DataFrame, a: list, b: list, distance_function) -> pd.DataFrame:
    """
    This function is used to calculate the distance between each pair of regions in list a and b, with the given distance
    function (e.g., euclidean)
    Input: df with feature values, list of regions in set a and b for distance calculation (if none, then compute all
    pairs of regions in df); distance function used to calculate the distance
    Output: distance matrix df
    """
    if a is not None:
        a_data = df.loc[a, :]
    else:
        a_data = df

    if b is not None:
        b_data = df.loc[b, :]
    else:
        b_data = df

    distance_matrix = pd.DataFrame(columns=b_data.index, index=a_data.index, dtype=float)
    for a_ind in a_data.index:
        for b_ind in b_data.index:
            distance_matrix.loc[a_ind, b_ind] = distance_function(a_data.loc[a_ind], b_data.loc[b_ind])

    return distance_matrix


def generate_feature_for_dist_compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to create feature(s) needed to compute the similarity scores
    Input: raw df
    Output: df with created feature(s) (e.g., revenue by region)
    """
    return df.groupby(['region_id'])['revenue'].sum().to_frame()


def compute_similarity(df_revenue_by_region: pd.DataFrame, a: list, b: list) -> float:
    """
    This function is used to compute the overall similarity score between set a and b
    Input: df with features used to calculate similarity score, list of regions in set a, list of regions in set b
    Output: overall similarity score
    Note: a,b should not have overlap, the implementation of this function requires providing a distance measurement
    between two data points.
    """
    distance_matrix = compute_distance_matrix_internal(df_revenue_by_region, a, b, scipy.spatial.distance.euclidean)
    min_value_a_b = distance_matrix.min(axis=1)
    min_value_b_a = distance_matrix.min(axis=0)
    similarity_score = min_value_a_b.sum()/len(a) + min_value_b_a.sum()/len(b)
    return similarity_score


def get_similar_sets(df_revenue_by_region: pd.DataFrame) -> list:
    """
    This function is used assign regions to set a and b; the logic is as follows: for each region, we find its nearest
    neighbor (i.e., most similar revenue) so we end up with 100 pairs of regions; then we sort the pairs by the distance
    (i.e., diff. in revenue for regions in each pair); finally we begin the assignment, we begin with the pair that has
    the min. distance, and assign 1st regions to set a, and its neighbor to set b, then we move on the the next pair and
    continue the assignment, until we reach 10 regions for set a and b
    Input: df with features used to calculate similarity score
    Output: list of regions in set a, list of regions in set b
    """

    df_dist = compute_distance_matrix_internal(df_revenue_by_region, None, None, scipy.spatial.distance.euclidean)
    print(df_dist.head())
    df_dist['Min_val'] = df_dist[df_dist != 0].min(axis=1)
    df_dist['Min_col'] = df_dist[df_dist != 0].idxmin(axis=1)
    df_nn = df_dist[['Min_col', 'Min_val']]
    df_nn.sort_values(by='Min_val')

    A = set([])
    B = set([])
    for index, row in df_nn.iterrows():
        if len(A) == 10:
            break
        else:
            if index in B:
                continue
            else:
                A.add(index), B.add(row['Min_col'])
    return A, B


df = pd.read_parquet('snapshot.parquet', engine='pyarrow')
df_revenue_by_region = generate_feature_for_dist_compute(df)
A, B = get_similar_sets(df_revenue_by_region)
similarity_score = compute_similarity(df, list(A), list(B))
print(f"A: {A} B: {B}")
print(f"Similarity score: {similarity_score}")