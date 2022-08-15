# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_rknn_net)

mmdeploy_add_module(${PROJECT_NAME} rknn_net.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
        $<BUILD_INTERFACE:${RKNN_DIR}>)
