// Copyright (C) 2016 Toyota Motor Corporation
#include "hsrb_pseudo_endeffector_position_controller.hpp"
#include <map>
#include <string>
#include <std_msgs/Bool.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <tmc_manipulation_types_bridge/manipulation_msg_convertor.hpp>

namespace {
// 数値IK許容誤差
const double kDefaultIKDelta = 1.0e-3;
// 速度制御時の補完時間
const double kDefaultVelocityDuration = 0.5;
// 数値IK最大繰り返し回数
const int32_t kMaxItrIK = 1000;
// 数値IK許容変動
const double kIKConvergeThreshold = 1.0e-10;
// Trajectoryの開始時間[50ms]
// 短くすると
// 「Unexpected error: No trajectory defined at current time. Please contact the package maintainer」
// というROS_ERRORが発生する
const double kTrajectoryStartDelay = 0.05;
// 指令が途切れたと判断するしきい値[sec]
const double kDefaultDiscontinuousPeriod = 0.5;

// Warning付きパラメータ読み込み関数
template<typename T>
void ReadParam(ros::NodeHandle* nh,
               const char* param_name,
               const T& default_value,
               T& value) {
  if (!nh->getParam(param_name, value)) {
    ROS_WARN_STREAM("Parameter " << param_name << " is not set. "
                    "Use default value " << default_value);
    value = default_value;
  }
}

// TwistをTransformにする(durationをかける)
void TwistToTransform(const geometry_msgs::Twist& twist,
                      double duration,
                      Eigen::Affine3d& transform_out) {
  Eigen::Quaterniond zero = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond angular =
      Eigen::AngleAxisd(twist.angular.x, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(twist.angular.y, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(twist.angular.z, Eigen::Vector3d::UnitZ());
  // angluarはslerpで補間
  transform_out = zero.slerp(duration, angular);
  // linearはdurationをかけるだけ
  transform_out.translation() <<
      twist.linear.x * duration,
      twist.linear.y * duration,
      twist.linear.z * duration;
}

void PoseToTransform(const geometry_msgs::Pose& pose, Eigen::Affine3d& transform_out) {
    // Extract position components
    double x = pose.position.x;
    double y = pose.position.y;
    double z = pose.position.z;

    // Extract orientation components
    double qx = pose.orientation.x;
    double qy = pose.orientation.y;
    double qz = pose.orientation.z;
    double qw = pose.orientation.w;

    // Construct the translation matrix
    Eigen::Translation3d translation(x, y, z);

    // Construct the rotation matrix from quaternion
    Eigen::Quaterniond rotation(qw, qx, qy, qz);
    Eigen::Transform<double, 3, Eigen::Affine> transform(rotation);

    // Combine translation and rotation to create the Affine transformation
    transform_out = translation * transform;
}

}  // anonymous namespace

namespace hsrb_pseudo_endeffector_position_controller {

using tmc_robot_kinematics_model::Tarp3Wrapper;
using tmc_robot_kinematics_model::IKRequest;
using tmc_robot_kinematics_model::IKResult;
using tmc_robot_kinematics_model::NumericIKSolver;
using tmc_manipulation_types::JointState;

PseudoEndeffectorPositionController::PseudoEndeffectorPositionController(ros::NodeHandle* nh,
                                                                         ros::NodeHandle* nh_private)
    : last_command_stamp_(0), last_command_frame_("") {
  // パラメータ読み込み
  double ik_delta;
  std::string robot_description;
  if (!nh->getParam("robot_description", robot_description)) {
    ROS_FATAL("Failed to get parameter /robot_description");
    exit(EXIT_FAILURE);
  }
  ReadParam(nh_private, "ik_delta", kDefaultIKDelta, ik_delta);
  ReadParam(nh_private, "velocity_duration", kDefaultVelocityDuration, velocity_duration_);
  ReadParam(nh_private, "discontinuous_period", kDefaultDiscontinuousPeriod, discontinuous_period_);

  // ロボットモデル作成
  endeffector_frame_name_ = "hand_palm_link";
  const char* names[] = {"arm_lift_joint",
                         "arm_flex_joint",
                         "arm_roll_joint",
                         "wrist_flex_joint",
                         "wrist_roll_joint"};
  use_joints_ = NameSeq(names, names + 5);

  //ik_arm_weights_ = GetWeightParameter("ik_arm_weights", use_joints_.size());

  ik_arm_weights_.resize(5);
  ik_arm_weights_ << 1.0, 100.0, 1.0, 1.0, 1.0;
  // 台車はodom_x/y/tの3自由度を想定
  //ik_base_weights_ = GetWeightParameter("ik_base_weights", 3);
  ik_base_weights_.resize(3);
  ik_base_weights_ << 1.0, 1.0, 100.0;

  // Subscriber作成
  command_position_sub_ =
      nh_private->subscribe<geometry_msgs::PoseStamped>(
          "command_position", 1, &PseudoEndeffectorPositionController::CommandPositionCallback, this);
  command_position_with_base_sub_ =
      nh_private->subscribe<geometry_msgs::PoseStamped>(
          "command_position_with_base", 1, &PseudoEndeffectorPositionController::CommandPositionWithBaseCallback, this);
  joint_states_sub_ = nh->subscribe<sensor_msgs::JointState>(
      "joint_states", 1, &PseudoEndeffectorPositionController::JointStatesCallback, this);
  odom_sub_ = nh->subscribe<nav_msgs::Odometry>(
      "odom", 1, &PseudoEndeffectorPositionController::OdomCallback, this);

  // Publisher作成
  arm_command_pub_ = nh->advertise<trajectory_msgs::JointTrajectory>(
      "arm_trajectory_controller/command", 1, true);
  base_command_pub_ = nh->advertise<trajectory_msgs::JointTrajectory>(
      "omni_base_controller/command", 1, true);
  success_pub_ = nh_private->advertise<std_msgs::Bool>(
      "success", 1, true);

  robot_.reset(new Tarp3Wrapper(robot_description));
  ik_solver_.reset(new NumericIKSolver(IKSolver::Ptr(), robot_,
                                       kMaxItrIK, ik_delta, kIKConvergeThreshold));
}

// 最新のjoint_states, odomからロボットの状態を更新する
bool PseudoEndeffectorPositionController::UpdateJointState() {
  if (!latest_joint_states_ || !latest_odom_) {
    ROS_ERROR_THROTTLE(60, "joint_states or odom is not subscribed yet");
    return false;
  }

  // コールバックで中身更新されないようにポインタをおさえる
  sensor_msgs::JointState::ConstPtr latest_joint_states = latest_joint_states_;
  nav_msgs::Odometry::ConstPtr latest_odom = latest_odom_;

  // joint_statesの中身をJointState型に変換
  JointState joint_state;
  tmc_manipulation_types_bridge::JointStateMsgToJointState(*latest_joint_states,
                                                           joint_state);
  // odomの内容をEigen::Affine3d型に変換
  Eigen::Affine3d origin_to_base;
  origin_to_base = Eigen::Quaterniond(latest_odom->pose.pose.orientation.w,
                                      latest_odom->pose.pose.orientation.x,
                                      latest_odom->pose.pose.orientation.y,
                                      latest_odom->pose.pose.orientation.z);
  origin_to_base.translation() <<
      latest_odom->pose.pose.position.x,
      latest_odom->pose.pose.position.y,
      latest_odom->pose.pose.position.z;

  // 更新
  robot_->SetRobotTransform(origin_to_base);
  robot_->SetNamedAngle(joint_state);
  return true;
}

// 現在のロボットの状態からcommandを発行する
void PseudoEndeffectorPositionController::PublishJointCommand(const ros::Time& stamp,
                                                      double duration) const {
  // robot_の姿勢は更新済みとする
  JointState joint_state_out = robot_->GetNamedAngle(use_joints_);
  Eigen::Affine3d origin_to_base_out = robot_->GetRobotTransform();
  int32_t num_joints = use_joints_.size();

  // arm用JointTrajectoryを作成
  trajectory_msgs::JointTrajectory arm_trajectory;
  arm_trajectory.joint_names = use_joints_;
  arm_trajectory.points.resize(1);
  arm_trajectory.points[0].positions.resize(num_joints);
  arm_trajectory.points[0].velocities.resize(num_joints);
  for (int i = 0; i < num_joints; ++i) {
    arm_trajectory.points[0].positions[i] = joint_state_out.position[i];
    arm_trajectory.points[0].velocities[i] = 0.0;
  }
  arm_trajectory.points[0].time_from_start = ros::Duration(duration);

  // base用JointTrajectoryを作成
  trajectory_msgs::JointTrajectory base_trajectory;
  base_trajectory.joint_names.push_back("odom_x");
  base_trajectory.joint_names.push_back("odom_y");
  base_trajectory.joint_names.push_back("odom_t");
  base_trajectory.points.resize(1);
  base_trajectory.points[0].positions.resize(3);
  base_trajectory.points[0].velocities.resize(3);
  Eigen::Vector3d origin_to_base_rpy = origin_to_base_out.rotation().eulerAngles(0, 1, 2);
  base_trajectory.points[0].positions[0] = origin_to_base_out.translation()[0];
  base_trajectory.points[0].positions[1] = origin_to_base_out.translation()[1];
  base_trajectory.points[0].positions[2] = origin_to_base_rpy[2];
  base_trajectory.points[0].velocities[0] = 0.0;
  base_trajectory.points[0].velocities[1] = 0.0;
  base_trajectory.points[0].velocities[2] = 0.0;
  base_trajectory.points[0].time_from_start = ros::Duration(duration);

  // publish
  arm_trajectory.header.stamp = stamp;
  base_trajectory.header.stamp = stamp;
  arm_command_pub_.publish(arm_trajectory);
  base_command_pub_.publish(base_trajectory);
}

void CalcOriginToNextEnd(const Eigen::Affine3d& origin_to_end,
                         const Eigen::Affine3d& origin_to_frame,
                         const Eigen::Affine3d& transform,
                         Eigen::Affine3d& dst_origin_to_next_end) {
  // endeffector -> 基準フレームへの座標変換
  Eigen::Affine3d end_to_frame = origin_to_end.inverse() * origin_to_frame;
  // 原点を合わせる
  end_to_frame.translation() << 0.0, 0.0, 0.0;
  // 基準フレーム原点のtransformをendeffector原点に変換する
  Eigen::Affine3d end_on_transform = end_to_frame * transform * end_to_frame.inverse();
  // 次のendeffectorの位置を求める
  dst_origin_to_next_end = origin_to_end * end_on_transform;
}

// 手先移動させた後のロボットの状態を演算する
bool PseudoEndeffectorPositionController::CalcNextState(
    const tmc_manipulation_types::BaseMovementType& base_type,
    const Eigen::Affine3d& origin_to_next_end) {
  IKRequest ik_request(base_type);
  ik_request.use_joints = use_joints_;
  ik_request.frame_to_end = Eigen::Affine3d::Identity();
  ik_request.frame_name = endeffector_frame_name_;
  ik_request.origin_to_base = robot_->GetRobotTransform();
  ik_request.initial_angle = robot_->GetNamedAngle(use_joints_);
  ik_request.ref_origin_to_end = origin_to_next_end;
  if (base_type == tmc_manipulation_types::kRotationZ) {
    ik_request.weight.resize(use_joints_.size() + 1);
    ik_request.weight.head(use_joints_.size()) = ik_arm_weights_;
    ik_request.weight.tail(1) = ik_base_weights_.tail(1);
  } else if (base_type == tmc_manipulation_types::kPlanar) {
    ik_request.weight.resize(use_joints_.size() + 3);
    ik_request.weight.head(use_joints_.size()) = ik_arm_weights_;
    ik_request.weight.tail(3) = ik_base_weights_;
  }

  JointState joint_state_out;
  Eigen::Affine3d origin_to_end_out;
  bool result = ik_solver_->Solve(ik_request,
                                  joint_state_out,
                                  origin_to_end_out);
  if (result != tmc_robot_kinematics_model::kSuccess) {
    ROS_WARN_STREAM("ik fail");
    ROS_WARN_STREAM(endeffector_frame_name_);
    return false;
  }
  return true;
}

// position系コールバック共通処理
void PseudoEndeffectorPositionController::PositionCallback(
    const geometry_msgs::PoseStamped::ConstPtr& command,
    const tmc_manipulation_types::BaseMovementType& base_type) {
  boost::mutex::scoped_lock lock(lock_);
  if (!UpdateJointState()) {
    return;
  }

  Eigen::Affine3d transform;
  PoseToTransform(command->pose, transform);

  bool result = CalcNextState(base_type, transform);
  ros::Time stamp = ros::Time::now() + ros::Duration(kTrajectoryStartDelay);
  PublishJointCommand(stamp, velocity_duration_);
  // if (result) {
  //   // commandと同期して発行したいトピックはこのstampを使用する(現状未使用)
  //   ros::Time stamp = ros::Time::now() + ros::Duration(kTrajectoryStartDelay);
  //   PublishJointCommand(stamp, velocity_duration_);
  // } else {
  //   ROS_DEBUG_THROTTLE(10, "IK failed");
  // }
  PublishIsSuccess(result);
}

// 成否を発行する
void PseudoEndeffectorPositionController::PublishIsSuccess(bool result) {
  std_msgs::Bool success;
  success.data = result;
  success_pub_.publish(success);
}


// command_positionのコールバック
void PseudoEndeffectorPositionController::CommandPositionCallback
(const geometry_msgs::PoseStamped::ConstPtr& command) {
  PositionCallback(command, tmc_manipulation_types::kRotationZ);
}

// command_position_with_baseのコールバック
void PseudoEndeffectorPositionController::CommandPositionWithBaseCallback
(const geometry_msgs::PoseStamped::ConstPtr& command) {
  PositionCallback(command, tmc_manipulation_types::kPlanar);
}

Eigen::VectorXd PseudoEndeffectorPositionController::GetWeightParameter(const std::string& parameter_name, uint32_t dof) {
  // const auto ik_weights_vec = GetParameter<std::vector<double>>(parameter_name, {});
  Eigen::VectorXd ik_weights_eigen = Eigen::VectorXd::Ones(dof);
  //if (ik_weights_vec.size() != dof) {
    return ik_weights_eigen;
  //}
  //for (auto x : ik_weights_vec) {
  //  if (x <= std::numeric_limits<double>::min()) {
  //    return ik_weights_eigen;
  //  }
  // }
  //ik_weights_eigen = Eigen::Map<const Eigen::VectorXd>(&ik_weights_vec[0], ik_weights_vec.size());
  //return ik_weights_eigen;
}

}  // namespace hsrb_pseudo_endeffector_position_controller

int main(int argc, char* argv[]) {
  try {
    ros::init(argc, argv, "hsrb_pseudo_endeffector_position_controller_node");
    ros::NodeHandle nh("");
    ros::NodeHandle nh_private("~");
    hsrb_pseudo_endeffector_position_controller::PseudoEndeffectorPositionController
        hsrb_pseudo_endeffector_position_controller_node(&nh, &nh_private);
    ros::spin();
  } catch(const std::exception& ex) {
    ROS_ERROR("Exception occurred. (%s)", ex.what());
    exit(EXIT_FAILURE);
  }
  return EXIT_SUCCESS;
}
