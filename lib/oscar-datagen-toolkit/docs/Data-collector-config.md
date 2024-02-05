The **oscar_data_saver** tool use as input a YAML configuration file that has the following options:

## Context

The _context_ section is composed by those parameters that describes the CARLA simulation.

- Keyword: `context`.
- Object: `oscar_datagen_tools.simulation.Context`.

### Client parameters

Parameters that configure the CARLA client connection.

- Keyword: `client_params`.
- Object: `oscar_datagen_tools.simulation.parameters.ClientParameters`.

| Option    | Type    | Default value                          | Description                                                                             |
| --------- | ------- | -------------------------------------- | --------------------------------------------------------------------------------------- |
| `host`    | `str`   | "127.0.0.1"                            | IP address of the CARLA server.                                                         |
| `port`    | `int`   | 2000                                   | Port number of the CARLA server.                                                        |
| `timeout` | `float` | 5.0                                    | Duration (in seconds) or time limit for the timeout.                                    |
| `retry`   | `int`   | 5                                      | Max number of time the client will try to establish a connection with the CARLA server. |
| `seed`    | `int`   | Random number between 1 and "ffffffff" | Random seed number.                                                                     |

### Simulation parameters

Parameters that configure the simulation features.

- Keyword: `simulation_params`.
- Object: `oscar_datagen_tools.simulation.parameters.SimulationParameters`.

| Option                 | Type   | Default value | Description                                                                                                                                           |
| ---------------------- | ------ | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `traffic_manager_port` | `int`  | 8000          | Port number for the Traffic Manager in the CARAL server.                                                                                              |
| `respawn`              | `bool` | True          | Enable the respawn feature in the Traffic Manager object.                                                                                             |
| `townmap`              | `str`  | "Town01"      | Map to load on the simulation.                                                                                                                        |
| `warmup`               | `int`  | 150           | Number of steps that the simulation performs before start the actual data collection. This is useful to give time to the simulator to render the map. |
| `interval`             | `int`  | 1             | Number of simulation steps needed to capture the next sample.                                                                                         |

### Synchronization parameters

Parameters that configure the synchronization behavior.

- Keyword: `sync_params`.
- Object: `oscar_datagen_tools.simulation.parameters.SyncParameters`.

| Option    | Type    | Default value | Description                        |
| --------- | ------- | ------------- | ---------------------------------- |
| `fps`     | `float` | 0.2           | Simulation frame rate.             |
| `timeout` | `float` | 2.0           | Max duration to get sensor's data. |

### Weather parameters

Parameters that configure the weather conditions.

- Keyword: `weather_params`.
- Object: `carla.WeatherParameters`.

To see more information about the weather parameters, visit the [carla.WeatherParameters](https://carla.readthedocs.io/en/latest/python_api/#carlaweatherparameters) documentation.

## Actor's generator

Object that is in charge of the random generation of walkers and vehicles.

- Keyword: `actors_generator`.
- Object: `oscar_datagen_tools.simulation.actors.ActorsGenerator`.

| Option    | Type  | Default value              | Description                                                                                                                |
| --------- | ----- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `_actor_` | `str` | User must provide a value. | Type of actors to spawn: `oscar_datagen_tools.simulation.actors.Walker` or `oscar_datagen_tools.simulation.actors.Vehicle` |
| `seed`    | `int` | `None`                     | Generator seed to make the process deterministic.                                                                          |
| `count`   | `int` | User must provide a value. | Number of walkers/vehicles to sapwn.                                                                                       |

## Spawn actors

In this section is specified a list of actors that the user wants to spawn. This section is usually used to add cameras to de simulator.

- keyword: `spawn_actors`.
- Object: `list`.

> For more information of how to create the different types of actors, check the [actor's wiki page](https://github.com/intel-sandbox/carla-datagen-toolkit/wiki/Actors).

## Number of frames

To specify the max number of frames to capture use:

- Keyword: `max_frames`.
- Object: `int`.
