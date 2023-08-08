The following is an example of the structure of the configuration file for the `oscar_data_saver` tool:

```yaml
defaults:
  - _self_
  - paths: default.yaml
  - hydra: default.yaml

context:
  _target_: oscar_datagen_tools.simulation.Context
  client_params:
    _target_: oscar_datagen_tools.simulation.parameters.ClientParameters
    host: 127.0.01
    port: 2000
    timeout: 10.0
    retry: 10
    seed: 50
  simulation_params:
    _target_: oscar_datagen_tools.simulation.parameters.SimulationParameters
    traffic_manager_port: 8000
    respawn: true
    townmap: Town10HD
    warmup: 250
    interval: 1
  sync_params:
    _target_: oscar_datagen_tools.simulation.parameters.SyncParameters
    fps: 30.0
    timeout: 2.0
  weather_params:
    _target_: carla.WeatherParameters
    cloudiness: 0.0
    precipitation: 0.0
    precipitation_deposits: 0.0
    wind_intensity: 0.0
    sun_azimuth_angle: 0.0
    sun_altitude_angle: 10.0
    fog_density: 0.0
    fog_distance: 0.0
    wetness: 0.0

max_frames: 10

actors_generator:
  _target_: oscar_datagen_tools.simulation.actors.ActorsGenerator
  number_of_vehicles: 10
  number_of_walkers: 30

spawn_actors:
  - _target_: oscar_datagen_tools.simulation.actors.Camera
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
      num_waypoints: 4
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

To consult the available options and its default values, run:

```bash
./scritps/collect.sh --help
```

with the following output:
```bash
Collect new data from a CARLA sim server.
Usage: ./scritps/collect.sh [--scale=NUM] [CONFIGURATION-FILE]
oscar_data_saver is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

context: default
paths: default
simulation: mobile_sensors, pedestrian_spawn, sensors_hero, static_sensors, tracking


== Config ==
Override anything in the config (foo.bar=value)

context:
  _target_: oscar_datagen_tools.simulation.Context
  client_params:
    _target_: oscar_datagen_tools.simulation.parameters.ClientParameters
    host: 127.0.01
    port: 2000
    timeout: 5.0
    retry: 10
    seed: 30
  simulation_params:
    _target_: oscar_datagen_tools.simulation.parameters.SimulationParameters
    traffic_manager_port: 8000
    respawn: true
    townmap: Town10HD
  sync_params:
    _target_: oscar_datagen_tools.simulation.parameters.SyncParameters
    fps: 30.0
    timeout: 2.0
  weather_params:
    _target_: carla.WeatherParameters
    cloudiness: 0.0
    precipitation: 0.0
    precipitation_deposits: 0.0
    wind_intensity: 0.0
    sun_azimuth_angle: 0.0
    sun_altitude_angle: 10.0
    fog_density: 0.0
    fog_distance: 0.0
    wetness: 0.0
paths:
  now_dir: ${now:%Y-%m-%d}_${now:%H-%M-%S-%f}
  root_dir: ${oc.env:PROJECT_ROOT}
  data_dir: ${paths.root_dir}/data/SENSOR_TYPE/${paths.now_dir}
  log_dir: ${paths.root_dir}/logs/${paths.now_dir}
  output_dir: ${hydra:runtime.output_dir}
  work_dir: ${hydra:runtime.cwd}
max_frames: 100
actors_generator:
  _target_: oscar_datagen_tools.simulation.actors.ActorsGenerator
  number_of_vehicles: 30
  number_of_walkers: 60
spawn_actors:
- _target_: oscar_datagen_tools.simulation.actors.Camera
  image_size_x: 800
  image_size_y: 600
  speed: 1.0
  sensors_blueprints:
  - _target_: oscar_datagen_tools.simulation.blueprints.RGB
    gamma: 2.2
  - _target_: oscar_datagen_tools.simulation.blueprints.InstanceSegmentation
  motion_params:
    _target_: oscar_datagen_tools.simulation.parameters.MotionParameters
    sampling_resolution: 1
  transform:
    _target_: carla.Transform
    location:
      _target_: carla.Location
      x: -60
      'y': 1.0
      z: 2.0
    rotation:
      _target_: carla.Rotation
  destination_transform:
    _target_: carla.Transform
    location:
      _target_: carla.Location
      x: 45.0
      'y': 1.0
      z: 2.0


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```