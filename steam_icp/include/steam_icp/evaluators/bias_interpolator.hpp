#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace bias {

// class Variable {
//  public:
//   using Ptr = std::shared_ptr<Variable>;
//   using ConstPtr = std::shared_ptr<const Variable>;
//   using BiasType = Eigen::Matrix<double, 6, 1>;
//   static Ptr MakeShared(const Time& time, const Evaluable<BiasType>::Ptr& b) {
//     return std::make_shared<Variable>(time, b);
//   }

//   Variable(const Time& time, const Evaluable<BiasType>::Ptr& b) : time_(time), b_(b) {}

//   virtual ~Variable() = default;

//   const Time& time() const { return time_; }
//   const Evaluable<BiasType>::Ptr& bias() const { return b_; }

//  private:
//   Time time_;
//   const Evaluable<BiasType>::Ptr b_;
// };

class BiasInterpolator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
 public:
  using Ptr = std::shared_ptr<BiasInterpolator>;
  using ConstPtr = std::shared_ptr<const BiasInterpolator>;

  using InType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const Time& time, const Evaluable<InType>::ConstPtr& bias1, const Time& time1,
                        const Evaluable<InType>::ConstPtr& bias2, const Time& time2);
  BiasInterpolator(const Time& time, const Evaluable<InType>::ConstPtr& bias1, const Time& time1,
                   const Evaluable<InType>::ConstPtr& bias2, const Time& time2);
  bool active() const override;
  void getRelatedVarKeys(KeySet& keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const override;

 private:
  const Evaluable<InType>::ConstPtr bias1_;
  const Evaluable<InType>::ConstPtr bias2_;
  double psi_, lambda_;
};

}  // namespace bias
}  // namespace traj
}  // namespace steam