# Cases are built in Python, without configuration files

A case is a Python script that constructs the mesh, equations, terms, integrator and outputs, passing each object its dependencies explicitly. There is no YAML configuration and no string registry: in every earlier generation, the YAML file ended up read deep inside the numerics through a global context (`context.get().config.get('grid/N', int)`), which hid every dependency. A configuration-file front end may come later, as a thin layer that calls the same constructors.
