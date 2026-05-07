// Copyright (C) 2016 Toyota Motor Corporation
#ifndef HSRB_PSEUDO_ENDEFFECTOR_POSITION_CONTROLLER_HSRB_PSEUDO_ENDEFFECTOR_POSITION_CONTROLLER_HPP
#define HSRB_PSEUDO_ENDEFFECTOR_POSITION_CONTROLLER_HSRB_PSEUDO_ENDEFFECTOR_POSITION_CONTROLLER_HPP

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <tmc_manipulation_types/manipulation_types.hpp>
#include <tmc_robot_kinematics_model/numeric_ik_solver.hpp>
#include <tmc_robot_kinematics_model/tarp3_wrapper.hpp>

namespace hsrb_pseudo_endeffector_position_controller {

using tmc_robot_kinematics_model::IKSolver;
using tmc_robot_kinematics_model::IRobotKinematicsModel;
using tmc_manipulation_types::NameSeq;

// 擬似手先制御コントローラクラス
class PseudoEndeffectorPositionController {
 public:
  explicit PseudoEndeffectorPositionController(ros::NodeHandle* nh, ros::NodeHandle* nh_private);
  virtual ~PseudoEndeffectorPositionController() {}

 private:
  // PoseStamped系コールバック共通処理
  void PositionCallback(const geometry_msgs::PoseStamped::ConstPtr& command,
                        const tmc_manipulation_types::BaseMovementType& base_type);
  void CommandPositionCallback(const geometry_msgs::PoseStamped::ConstPtr& command);
  void CommandPositionWithBaseCallback(const geometry_msgs::PoseStamped::ConstPtr& command);
  // joint_statesコールバック
  void JointStatesCallback(const sensor_msgs::JointState::ConstPtr& state) {
    latest_joint_states_ = state;
  }
  // odomコールバック
  void OdomCallback(const nav_msgs::Odometry::ConstPtr& odom) {
    latest_odom_ = odom;
  }
  // 最新のjoint_states, odomからロボットの状態を更新する
  bool UpdateJointState();
  // 現在のロボットの状態からcommandを発行する
  void PublishJointCommand(const ros::Time& stamp,
                           double duration) const;
  // 手先移動させた後のロボットの状態を演算する
  bool CalcNextState(const tmc_manipulation_types::BaseMovementType& base_type,
                     const Eigen::Affine3d& origin_to_next_end);
  // 成否を発行する
  void PublishIsSuccess(bool result);

  ros::Subscriber command_position_sub_;
  ros::Subscriber command_position_with_base_sub_;
  ros::Subscriber joint_states_sub_;
  ros::Subscriber odom_sub_;
  ros::Publisher arm_command_pub_;
  ros::Publisher base_command_pub_;
  ros::Publisher success_pub_;

  sensor_msgs::JointState::ConstPtr latest_joint_states_;
  nav_msgs::Odometry::ConstPtr latest_odom_;

  // 排他制御ロック
  boost::mutex lock_;

  // endefffectorのフレーム名
  std::string endeffector_frame_name_;
  // 動作させる関節
  NameSeq use_joints_;

  Eigen::VectorXd ik_arm_weights_;
  Eigen::VectorXd ik_base_weights_;

  // ロボットモデル
  IRobotKinematicsModel::Ptr robot_;
  // IKソルバ
  IKSolver::Ptr ik_solver_;

  // 速度制御時の補間時間
  double velocity_duration_;

  // EndEffectorの姿勢
  Eigen::Affine3d origin_to_end_;
  // 指令値の基準フレームの姿勢
  Eigen::Affine3d origin_to_base_;
  // 前回の指令値
  geometry_msgs::Twist last_command_value_;
  // 前回の指令が飛んできた時刻
  ros::Time last_command_stamp_;
  // 前回の指令での基準フレーム
  std::string last_command_frame_;
  // 連続的な指令が途切れたと判断するしきい値[sec]
  double discontinuous_period_;

  Eigen::VectorXd GetWeightParameter(const std::string& parameter_name, uint32_t dof);
};
}  // namespace hsrb_pseudo_endeffector_position_controller

#endif  // HSRB_PSEUDO_ENDEFFECTOR_POSITION_CONTROLLER_HSRB_PSEUDO_ENDEFFECTOR_POSITION_CONTROLLER_HPP
