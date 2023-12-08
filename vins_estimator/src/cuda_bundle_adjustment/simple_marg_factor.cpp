#include "simple_marg_factor.h"

namespace VINS_FUSION_CUDA_BA {

void SimpleMargFactor::Construct(
    MarginalizationInfo* marg_info_ptr
) {
    n = marg_info_ptr->n;
    m = marg_info_ptr->m;

    linearized_jacobian = marg_info_ptr->linearized_jacobians;
    linearized_residual = marg_info_ptr->linearized_residuals;

    keep_block_idx  = marg_info_ptr->keep_block_idx;
    keep_block_size = marg_info_ptr->keep_block_size;
    keep_block_data = marg_info_ptr->keep_block_data;
}

void SimpleMargFactor::Clear() {
    valid = false;

    n = -1;
    m = -1;

    linearized_jacobian.setZero();
    linearized_residual.setZero();

    keep_block_idx.clear();
    keep_block_size.clear();
    keep_block_data.clear();
    keep_block_name.clear();

    src_pos.clear();
    dst_pos.clear();

    jac_reordered.setZero();
    jac_padded.setZero();
    res_padded.setZero();
}

void SimpleMargFactor::PadJacAndRes(int num_key_frames) {
    assert(keep_block_idx.size() == keep_block_size.size());
    assert(keep_block_idx.size() == keep_block_data.size());
    assert(keep_block_idx.size() == keep_block_name.size());

    for(int i = 0; i < keep_block_idx.size(); i++) {
        src_pos.emplace_back(
            (keep_block_idx[i] - m),
            (keep_block_size[i] == 7 ? 6 : keep_block_size[i])
        );
    }

    for(const auto& elem : keep_block_name) {
        int start_idx = 0;
        int end_idx = elem.find("[");
        std::string type = elem.substr(start_idx, end_idx - start_idx);

        start_idx = end_idx + 1;
        end_idx = elem.find("]");
        std::string id_str = elem.substr(start_idx, end_idx - start_idx);
        int id = atoi(id_str.c_str());

        if(type == "para_Pose") {
            dst_pos.emplace_back( (6 * 2 + 1 + 15 * id) , 6 );
        } else if (type == "para_SpeedBias") {
            dst_pos.emplace_back( (6 * 2 + 1 + 15 * id + 6) , 9 );
        } else if (type == "para_Td") {
            dst_pos.emplace_back( (6 * 2 + 1 * id) , 1 );
        } else if (type == "para_Ex_Pose") {
            dst_pos.emplace_back( (6 * id) , 6 );
        }
    }

    jac_reordered.resize(linearized_jacobian.rows(), (num_key_frames + 1) * 15 + 13);
    for(int i = 0; i < src_pos.size(); i++) {
        jac_reordered.middleCols(dst_pos[i].first, dst_pos[i].second) = linearized_jacobian.middleCols(src_pos[i].first, src_pos[i].second);
    }
    jac_padded.resize((num_key_frames + 1) * 15 + 13, (num_key_frames + 1) * 15 + 13); jac_padded.setZero();
    jac_padded.topRows(jac_reordered.rows()) = jac_reordered;

    res_padded.resize((num_key_frames + 1) * 15 + 13, 1); res_padded.setZero();
    res_padded.topRows(linearized_residual.rows()) = linearized_residual;
}

} // namespace VINS_FUSION_CUDA_BA