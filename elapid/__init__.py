"""User entrypoint to elapid"""

from elapid.features import (
    CategoricalTransformer,
    HingeTransformer,
    LinearTransformer,
    MaxentFeatureTransformer,
    ProductTransformer,
    QuadraticTransformer,
    ThresholdTransformer,
)
from elapid.geo import (
    annotate,
    apply_model_to_rasters,
    distance_weights,
    nearest_point_distance,
    sample_bias_file,
    sample_geoseries,
    sample_raster,
    sample_vector,
    stack_geometries,
    xy_to_geoseries,
    zonal_stats,
)
from elapid.models import EnsembleModel, MaxentModel, NicheEnvelopeModel
from elapid.stats import normalize_sample_probabilities
from elapid.train_test_split import BufferedLeaveOneOut, GeographicKFold, checkerboard_split
from elapid.utils import load_object, load_sample_data, save_object
