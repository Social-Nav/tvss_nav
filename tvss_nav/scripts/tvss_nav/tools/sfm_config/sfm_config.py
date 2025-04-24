#!/usr/bin/env python3

import roslibpy

# List of valid parameters that can be configured
VALID_PARAMS = {
    'max_lin_acc',     # Maximum acceleration in translation (m/s^2)
    'max_rot_acc',     # Maximum acceleration in rotation (rad/s^2)
    'max_lin_vel',     # Maximum linear velocity (m/s)
    'min_lin_vel',     # Minimum linear velocity (m/s)
    'max_rot_vel',     # Maximum angular velocity (rad/s)
    'min_rot_vel',     # Minimum angular velocity (rad/s)
    'sfm_goal_weight',     # Weight of the attraction force to the goal
    'sfm_obstacle_weight', # Weight of the repulsive force of the obstacles
    'sfm_people_weight',   # Weight of the repulsive force of the pedestrians
}

class SFMConfigError(Exception):
    """Custom exception for SFM configuration errors"""
    pass

def update_sfm_param(param_name: str, value: float, ros_client: roslibpy.Ros = None) -> bool:
    """
    Update a single SFM parameter using dynamic reconfigure via rosbridge.
    
    Args:
        param_name (str): Name of the parameter to update
        value (float): New value for the parameter
        ros_client (roslibpy.Ros, optional): Existing ROS connection to use. If None, creates new connection.
        
    Returns:
        bool: True if update was successful, False otherwise
        
    Raises:
        SFMConfigError: If parameter name is invalid or connection fails
    """
    if param_name not in VALID_PARAMS:
        raise SFMConfigError(f"Invalid parameter name: {param_name}. Must be one of {VALID_PARAMS}")
    
    try:
        # Use existing connection or create new one
        client = ros_client if ros_client else roslibpy.Ros(host='localhost', port=9090)
        if not ros_client:
            client.run()
        
        # Create service client for dynamic reconfigure
        service = roslibpy.Service(client, 
                                 '/rto/move_base_flex/SFMControllerROS/set_parameters',
                                 'dynamic_reconfigure/Reconfigure')
        
        # Prepare request
        request = {
            'config': {
                'doubles': [{'name': param_name, 'value': float(value)}],
                'bools': [],
                'ints': [],
                'strs': [],
                'groups': []
            }
        }
        
        # Call service
        result = service.call(request)
        
        # Clean up only if we created a new connection
        if not ros_client:
            client.terminate()
        
        return True
        
    except Exception as e:
        raise SFMConfigError(f"Failed to update parameter: {str(e)}")
    
    return False

if __name__ == '__main__':
    # Example usage
    try:
        success = update_sfm_param('max_lin_vel', 0.5)
        print(f"Parameter update {'succeeded' if success else 'failed'}")
    except SFMConfigError as e:
        print(f"Error: {e}")
