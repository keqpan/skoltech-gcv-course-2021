#!/bin/bash

OUT_PATH=$1
IMAGES_PATH=$2
COLMAP="colmap" # put path to your colmap

$COLMAP feature_extractor \
   --database_path $OUT_PATH/database.db \
   --image_path $IMAGES_PATH

$COLMAP exhaustive_matcher \
   --database_path $OUT_PATH/database.db

mkdir $OUT_PATH/sparse

$COLMAP mapper \
    --database_path $OUT_PATH/database.db \
    --image_path $IMAGES_PATH \
    --output_path $OUT_PATH/sparse

mkdir $OUT_PATH/dense

$COLMAP image_undistorter \
    --image_path $IMAGES_PATH \
    --input_path $OUT_PATH/sparse/0 \
    --output_path $OUT_PATH/dense \
    --output_type COLMAP \
    --max_image_size -1

$COLMAP patch_match_stereo \
    --workspace_path $OUT_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

$COLMAP stereo_fusion \
    --workspace_path $OUT_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $OUT_PATH/dense/fused.ply