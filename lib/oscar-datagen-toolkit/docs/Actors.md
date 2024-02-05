In the following wiki page is described the different actors that can be included in the CARLA simulation.

## Actor's base

All the actors use as base the `Actor` class, which has the following options:

| Option        | Type              | Default value | Description                 |
| ------------- | ----------------- | ------------- | --------------------------- |
| `transform`   | `carla.Transform` | None          | Position to spawn the actor |
| `attachments` | `List`            | `[]`          | List of attached actors     |

## Cameras

The camera actor is a generator of different types of sensors. All these sensors will have the same configuration (see `image_size_x`, `image_size_y`, `motion_blur_intensity` examples below).

The camera movements can be configure using the `motion_params` estructure. With this object is possible to determine the **sampling resolution** of the generated route of the camera and the **number of random waypoints** that can be used to create the route between the start and end points.

> If the `oscar_datagen_tools.simulation.actors.Location` and `oscar_datagen_tools.simulation.actors.Rotation` are used in the transforms options, they provide the following options to assign the values:
>
> - Set directly the `float` value.
> - Set a range of **Max** and **Min** range from where will be chosen a value randomly.

### Config representation

```YAML
    _target_: oscar_datagen_tools.simulation.actors.Camera
    modalities: [rgb, depth, instance_segmentation]
    image_size_x: 1280
    image_size_y: 960
    motion_blur_max_distortion: 0.0
    gamma: 2.2
    transform:
      _target_: carla.Transform
      location:
        _target_: carla.Location
        x: -50
        y: -15
        z: 40
      rotation:
        _target_: carla.Rotation
        roll: 0
        pitch: 310
        yaw: 82
```

### Options

| Option          | Type                                                               | Default value        | Description                                                                          |
| --------------- | ------------------------------------------------------------------ | -------------------- | ------------------------------------------------------------------------------------ |
| `motion_params` | `oscar_datagen_tools.simulation`<br>`.parameters.MotionParameters` | `MotionParameters()` | Parameters to configure the camera movement.                                         |
| `modalities`    | `List`                                                             | `[]`                 | List of modalities to include in the Camera (E.g. rgb, depth, instance_segmentation) |

> By default, the `Camera` object always includes an `InstanceSegmentation` sensor to have access to the corresponding scene labels.

#### MotionParameters options

| Option                | Type             | Default value            | Description                                                                                    |
| --------------------- | ---------------- | ------------------------ | ---------------------------------------------------------------------------------------------- |
| `sampling_resolution` | `int`            | 2                        | Distance between waypoints in the sensor's route.                                              |
| `lane_type`           | `carla.LaneType` | `carla.LaneType.Driving` | [CARLA lane type](https://carla.readthedocs.io/en/latest/python_api/#carlalanetype).           |
| `num_waypoints`       | `int`            | 0                        | Number of random waypoint to use between the start and end waypoints of the sensor's route.    |
| `points`              | `List`           | `[]`                     | Read-only property that provides the random waypoints specified in the `num_waypoints` option. |

## Walkers

These actors are a representation of the pedestrians in the simulation. To a `Walker` can be attached a `oscar_datagen_tools.simulation.actors.WalkerController` that move them along the map or a `Camera` that follows the actor during the simulation.

### Config representation

```YAML
    - _target_: oscar_datagen_tools.simulation.actors.Walker
    is_invincible: false
    speed: 1.4
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
        transform:
          _target_: carla.Transform
          location:
            _target_: carla.Location
            x: -91
            y: 150
            z: 0.6

      - _target_: oscar_datagen_tools.simulation.actors.Sensor
        name: camera.rgb
        image_size_x: 800
        image_size_y: 600
        gamma: 2.2
        transform:
          _target_: carla.Transform
          location:
            _target_: carla.Location
            x: -6.0
            z: 1.5
```

### Options

| Option          | Type    | Default value | Description           |
| --------------- | ------- | ------------- | --------------------- |
| `is_invincible` | `bool`  | False         | -                     |
| `speed`         | `float` | 0.0           | Walker's speed value. |

#### WalkerAIController options

| Option      | Type    | Default value | Description                     |
| ----------- | ------- | ------------- | ------------------------------- |
| `max_speed` | `float` | 0.0           | Max speed value for the walker. |

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

| Option           | Type   | Default value | Description         |
| ---------------- | ------ | ------------- | ------------------- |
| `color`          | `str`  | "255,0,0"     | RGB representation. |
| `sticky_control` | `bool` | False         | -                   |
| `terramechanics` | `bool` | False         | -                   |

## Patches

These actors represents green screen patches that can be placed as static elements in the simulation. These elements also have the possibility of draw an specific texture specified by the user.

### Config representation

```YAML
- _target_: oscar_datagen_tools.simulation.actors.Patch
    width: 12.25
    height: 12.15
    transform:
      _target_: carla.Transform
      location:
        _target_: carla.Location
        x: 88
        y: 47.5
        z: 6.175
      rotation:
        _target_: carla.Rotation
        roll: 90
        pitch: 0
        yaw: -90
    texture: ${paths.texture_dir}/patch.png
```

By default the texture images can be placed in `<ROOT_REPO>/data/texture` represented in the variable `paths.texture_dir`.

### Options

| Option    | Type    | Default value              | Description                |
| --------- | ------- | -------------------------- | -------------------------- |
| `width`   | `float` | User must provide a value. | Patch's width.             |
| `height`  | `float` | User must provide a value. | Patch's height.            |
| `texture` | `str`   | `""`                       | Path to the texture image. |
