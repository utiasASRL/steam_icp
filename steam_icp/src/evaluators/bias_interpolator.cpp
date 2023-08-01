#include "steam_icp/evaluators/bias_interpolator.hpp"

namespace steam {
namespace traj {
namespace bias {

BiasInterpolator::Ptr BiasInterpolator::MakeShared(const Time& time, const Variable::ConstPtr& knot1,
                                                   const Variable::ConstPtr& knot2) {
  return std::make_shared<BiasInterpolator>(const Time& time, const Variable::ConstPtr& knot1,
                                            const Variable::ConstPtr& knot2);
}

BiasInterpolator::BiasInterpolator(const Time& time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2)
    : knot1_(knot1), knot2_(knot2) {
  if (time < knot1->time() || time > knot2->time())
    throw std::runtime_error("time < knot1->time() || time > knot2->time()");
  const double tau = (time - knot1->time()).seconds();
  const double T = (knot2->time() - knot1->time()).seconds();
  const double ratio = tau / T;
  psi_ = ratio;
  lambda_ = 1 - ratio;
}

bool BiasInterpolator::active() const { return knot1_->bias()->active() || knot2->bias()->active(); }

void BiasInterpolator::getRelatedVarKeys(KeySet& keys) const {
  knot1_->bias()->getRelatedVarKeys(keys);
  knot2_->bias()->getRelatedVarKeys(keys);
}

auto BiasInterpolator::value() const -> OutType {
  const auto b1 = knot1_->bias()->value();
  const auto b2 = knot2_->bias()->value();
  return lambda_ * b1 + psi_ * b2;
}

auto BiasInterpolator::forward() const -> Node<OutType>::Ptr {
  const auto b1 = knot1_->bias()->forward();
  const auto b2 = knot2_->bias()->forward();
  OutType b = lambda_ * b1->value() + psi_ * b2->value();
  const auto node = Node<OutType>::MakeShared(b);
  node->addChild(b1);
  node->addChild(b2);
  return node;
}

void BiasInterpolator::backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const {
  if (!active()) return;
  if (knot1_->bias()->active()) {
    const auto b1_ = std::static_pointer_cast<Node<InType>>(node->at(0));
    knot1_->bias()->backward(lambda_ * lhs, b1_, jacs);
  }
  if (knot2_->bias()->active()) {
    const auto b2_ = std::static_pointer_cast<Node<InType>>(node->at(1));
    knot1_->bias()->backward(psi_ * lhs, b2_, jacs);
  }
}

}  // namespace bias
}  // namespace traj
}  // namespace steam