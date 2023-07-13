In the following wiki page is described the different actors that can be included in the CARLA simulation.

## Actor's base

All the actors use as base the `Actor` class, which has the following options:

| Option | Type |  Default value | Description |
| ------ | ---- | -------------- | ----------- |
| `transform` | `carla.Transform` | None | Position to spawn the actor |
| `destination_transform` | `carla.Transform` | None | Final position of the actor |
| `attachments` | `List` | `[]` | List of attached actors |
| `attachment_type` | `carla.AttachmentType` | `carla.AttachmentType.Rigid` | [CARLA attachments types](https://carla.readthedocs.io/en/latest/python_api/#carlaattachmenttype) |

> If the `transform` value is not provided, a random one is set.

> If the `destination_transform` is not provided, the `transform`'s value is assigned.

## Cameras

The camera actor is a compositions of different types of sensors. All these sensors will have the same configuration (see `image_size_x`, `image_size_y`, `motion_blur_intensity` examples bellow), but there are also sensor specific configurations that are set at the `sensor_blueprints` scope (see RGB's `gamma` example bellow).

The camera movements can be configure using the `motion_params` estructure. With this object is possible to determine the **sampling resolution** of the generated route of the camera and the **number of random waypoints** that can be used to create the route between the start and end points.

> If the `oscar_datagen_tools.simulation.actors.Location` and `oscar_datagen_tools.simulation.actors.Rotation` are used in the transforms options, they provide the following options to assign the values:
> * Set directly the `float` value.
> * Set a range of **Max** and **Min** range from where will be chosen a value randomly.

### Config representation
```YAML
    _target_: oscar_datagen_tools.simulation.actors.Camera
    image_size_x: 1280
    image_size_y: 960
    speed: 1.0
    motion_blur_intensity: 0.0
    motion_blur_max_distortion: 0.0
    sensors_blueprints:
      - _target_: oscar_datagen_tools.simulation.blueprints.Depth
      - _target_: oscar_datagen_tools.simulation.blueprints.RGB
        gamma: 2.2
      - _target_: oscar_datagen_tools.simulation.blueprints.InstanceSegmentation
    motion_params:
      _target_: oscar_datagen_tools.simulation.parameters.MotionParameters
      sampling_resolution: 10
      num_waypoints: 2
    transform:
      _target_: carla.Transform
      location:
        _target_: oscar_datagen_tools.simulation.actors.Location
        x: [-100.0, 100.0]
        y: [-100.0, 100.0]
        z: [20.0, 40.0]
      rotation:
        _target_: oscar_datagen_tools.simulation.actors.Rotation
        pitch: [-90.0, -45.0]
        yaw: [0, 360]
```

### Options
| Option | Type |  Default value | Description |
| ------ | ---- | -------------- | ----------- |
| `motion_params` | `oscar_datagen_tools.simulation`<br>`.parameters.MotionParameters` | `MotionParameters()` | Parameters to configure the camera movement. |
| `sensors_blueprints` | `List` | `[]` | List of sensor's blueprints. <br> - [RGB](https://github.com/intel-sandbox/carla-datagen-toolkit/blob/fc7aaa814d661ac2c016881508ebbca187d2ede4/oscar_datagen_tools/simulation/blueprints/sensor_blueprints.py#L35) <br> - [Depth](https://github.com/intel-sandbox/carla-datagen-toolkit/blob/fc7aaa814d661ac2c016881508ebbca187d2ede4/oscar_datagen_tools/simulation/blueprints/sensor_blueprints.py#L79) <br> - [Instance segmentation](https://github.com/intel-sandbox/carla-datagen-toolkit/blob/fc7aaa814d661ac2c016881508ebbca187d2ede4/oscar_datagen_tools/simulation/blueprints/sensor_blueprints.py#L73) |

> By default, the `Camera` object always includes an `InstanceSegmentation` sensor to have access to the corresponding scene labels.

#### MotionParameters options

| Option | Type |  Default value | Description |
| ------ | ---- | -------------- | ----------- |
| `sampling_resolution` | `int` | 2 | Distance between waypoints in the sensor's route. |
| `lane_type` | `carla.LaneType` | `carla.LaneType.Driving` | [CARLA lane type](https://carla.readthedocs.io/en/latest/python_api/#carlalanetype). |
| `num_waypoints` | `int` | 0 | Number of random waypoint to use between the start and end waypoints of the sensor's route. |
| `points` | `List` | `[]` | Read-only property that provides the random waypoints specified in the `num_waypoints` option. |

## Walkers

These actors are a representation of the pedestrians in the simulation. To a `Walker` can be attached a `oscar_datagen_tools.simulation.actors.WalkerController` that move them along the map or a `Camera` that follows the actor during the simulation.

### Config representation
```YAML
    - _target_: oscar_datagen_tools.simulation.actors.Walker
    role_name: "hero1"
    is_invincible: false
    speed: 1.4 # Between 1 and 2 m/s (default is 1.4 m/s).
    transform:
      _target_: carla.Transform
      location:
        _target_: carla.Location
        x: -91
        y: 170
        z: 0.6
      rotation:
        _target_: carla.Rotation
        yaw: -90.0
    attachments:
      - _target_: oscar_datagen_tools.simulation.actors.WalkerController
        max_speed: 1.4
        destination_transform:
          _target_: carla.Transform
          location:
            _target_: carla.Location
            x: -91
            y: 150
            z: 0.6

      - _target_: oscar_datagen_tools.simulation.actors.Camera
        image_size_x: 800
        image_size_y: 600
        sensors_blueprints:
          - _target_: oscar_datagen_tools.simulation.blueprints.RGB
            gamma: 2.2
          - _target_: oscar_datagen_tools.simulation.blueprints.Depth
          - _target_: oscar_datagen_tools.simulation.blueprints.InstanceSegmentation
        transform:
          _target_: carla.Transform
          location:
            _target_: carla.Location
            x: -6.0
            z: 1.5
```

### Options
| Option | Type |  Default value | Description |
| ------ | ---- | -------------- | ----------- |
| `is_invincible` | `bool` | False | - |
| `speed` | `float` | 0.0 | Walker's speed value. |

#### WalkerAIController options
| Option | Type |  Default value | Description |
| ------ | ---- | -------------- | ----------- |
| `max_speed` | `float` | 0.0 | Max speed value for the walker. |

## Vehicles

These actors are the representation of the different vehicles in the simulation. All these actors are controlled by the `TrafficManager` configured in the `Context` object.

### Config representation
```YAML
  - _target_: oscar_datagen_tools.simulation.actors.Vehicle
    role_name: "autopilot"
    color: "125,80,220" # RGB color
    transform:
      _target_: carla.Transform
      location:
        _target_: carla.Location
        x: -91
        y: 170
        z: 0.6
      rotation:
        _target_: carla.Rotation
        yaw: -90.0
```

### Options
| Option | Type |  Default value | Description |
| ------ | ---- | -------------- | ----------- |
| `color` | `str` | "255,0,0" | RGB representation. |
| `sticky_control` | `bool` | False | - |
| `terramechanics` | `bool` | False | - |