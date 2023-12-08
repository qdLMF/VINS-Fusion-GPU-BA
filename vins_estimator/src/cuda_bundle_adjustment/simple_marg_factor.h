#include <vector>
#include <string>

#include <eigen3/Eigen/Core>

#include "../factor/marginalization_factor.h"

namespace VINS_FUSION_CUDA_BA {

class SimpleMargFactor {
public :
    bool valid;

    int m = 0;
    int n = 0;

    std::vector<int> keep_block_idx;
    std::vector<int> keep_block_size;
    std::vector<double*> keep_block_data;
    std::vector<std::string> keep_block_name;

    Eigen::MatrixXd linearized_jacobian;
    Eigen::VectorXd linearized_residual;

    // ----------

    std::vector< std::pair<int, int> > src_pos;
    std::vector< std::pair<int, int> > dst_pos;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jac_reordered;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jac_padded;
    Eigen::Matrix<double, Eigen::Dynamic, 1> res_padded;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public : 
    void Construct(MarginalizationInfo* marg_info_ptr);
    void Clear();
    void PadJacAndRes(int num_key_frames);
};

} // namespace VINS_FUSION_CUDA_BA