#include "steam_icp/evaluators/bias_interpolator.hpp"

namespace steam {
namespace traj {
namespace bias {

BiasInterpolator::Ptr BiasInterpolator::MakeShared(const Time& time, const Evaluable<InType>::ConstPtr& bias1,
                                                   const Time& time1, const Evaluable<InType>::ConstPtr& bias2,
                                                   const Time& time2) {
  return std::make_shared<BiasInterpolator>(time, bias1, time1, bias2, time2);
}

BiasInterpolator::BiasInterpolator(const Time& time, const Evaluable<InType>::ConstPtr& bias1, const Time& time1,
                                   const Evaluable<InType>::ConstPtr& bias2, const Time& time2)
    : bias1_(bias1), bias2_(bias2) {
  if (time < time1 || time > time2) throw std::runtime_error("time < time1 || time > time2");
  const double tau = (time - time1).seconds();
  const double T = (time2 - time1).seconds();
  const double ratio = tau / T;
  psi_ = ratio;
  lambda_ = 1 - ratio;
}

bool BiasInterpolator::active() const { return bias1_->active() || bias2_->active(); }

void BiasInterpolator::getRelatedVarKeys(KeySet& keys) const {
  bias1_->getRelatedVarKeys(keys);
  bias2_->getRelatedVarKeys(keys);
}

auto BiasInterpolator::value() const -> OutType { return lambda_ * bias1_->value() + psi_ * bias2_->value(); }

auto BiasInterpolator::forward() const -> Node<OutType>::Ptr {
  const auto b1 = bias1_->forward();
  const auto b2 = bias2_->forward();
  OutType b = lambda_ * b1->value() + psi_ * b2->value();
  const auto node = Node<OutType>::MakeShared(b);
  node->addChild(b1);
  node->addChild(b2);
  return node;
}

void BiasInterpolator::backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const {
  if (!active()) return;
  if (bias1_->active()) {
    const auto b1_ = std::static_pointer_cast<Node<InType>>(node->at(0));
    bias1_->backward(lambda_ * lhs, b1_, jacs);
  }
  if (bias2_->active()) {
    const auto b2_ = std::static_pointer_cast<Node<InType>>(node->at(1));
    bias2_->backward(psi_ * lhs, b2_, jacs);
  }
}

}  // namespace bias
}  // namespace traj
}  // namespace steam