from aitviewer.viewer import Viewer
from aitviewer.scene.node import Node
from typing import Tuple
from aitviewer.remote.message import Message
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.volume import Volume
from lib.libmise import mise

class MiseMeshes(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.frames = []
        self.mc: Volume = None
    
    def add_frame(self, data):
        self.frames.append(data)
        self.n_frames = len(self.frames)
    
    def on_frame_update(self):
        if len(self.frames) > 0:
            data = self.frames[self.current_frame_id]

            r = data["res"]
            volume = mise.to_dense(data["points"], data["values"], data["res"])

            if self.mc is None:
                self.mc = Volume(volume, max_vertices=450000, max_triangles=600000, invert_normals=True, name=f"MC mesh {r}x{r}x{r}")
                self.add(self.mc)
            else:
                self.mc.volume = volume

class CustomViewer(Viewer):
    def __init__(self, meshes: MiseMeshes, **kwargs):
        self.m = meshes
        super().__init__(**kwargs)

    def process_message(self, type: Message, remote_uid: int, args: list, kwargs: dict, client: Tuple[str, str]):
        if type == Message.USER_MESSAGE:
            self.m.add_frame(kwargs["data"])
            self.scene.next_frame()
        return super().process_message(type, remote_uid, args, kwargs, client)


C.update_conf({'server_enabled': True})
meshes = MiseMeshes()

v = CustomViewer(meshes)
v.scene.add(meshes)
v.run()
