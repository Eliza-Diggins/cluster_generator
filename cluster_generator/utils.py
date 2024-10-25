# """
# Utility functions for basic functionality of the py:module:`cluster_generator` package.
# """
# import multiprocessing
# from typing import Union
#
# import h5py
# import numpy as np
# from scipy.integrate import quad
#
# from cluster_generator.utilities.typing import parse_prng
#
# # -- Utility functions -- #
# _truncator_function = lambda a, r, x: 1 / (1 + (x / r) ** a)
#
#
# class TimeoutException(Exception):
#     def __init__(self, msg="", func=None, max_time=None):
#         self.msg = f"{msg} -- {str(func)} -- max_time={max_time} s"
#
#
# def _daemon_process_runner(*args, **kwargs):
#     # Runs the function specified in the kwargs in a daemon process #
#
#     send_end = kwargs.pop("__send_end")
#     function = kwargs.pop("__function")
#
#     try:
#         result = function(*args, **kwargs)
#     except Exception as e:
#         send_end.send(e)
#         return
#
#     send_end.send(result)
#
#
# def time_limit(function, max_execution_time, *args, **kwargs):
#     """
#     Assert a maximal time limit on functions with potentially problematic / unbounded execution times.
#
#     .. warning::
#
#         This function launches a daemon process.
#
#     Parameters
#     ----------
#     function: callable
#         The function to run under the time limit.
#     max_execution_time: float
#         The maximum runtime in seconds.
#     args:
#         arguments to pass to the function.
#     kwargs: optional
#         keyword arguments to pass to the function.
#
#     """
#     import time
#
#     from tqdm import tqdm
#
#     recv_end, send_end = multiprocessing.Pipe(False)
#     kwargs["__send_end"] = send_end
#     kwargs["__function"] = function
#
#     tqdm_kwargs = {}
#     for key in ["desc"]:
#         if key in kwargs:
#             tqdm_kwargs[key] = kwargs.pop(key)
#
#     N = 1000
#
#     p = multiprocessing.Process(target=_daemon_process_runner, args=args, kwargs=kwargs)
#     p.start()
#
#     for _ in tqdm(
#             range(N),
#             **tqdm_kwargs,
#             bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining} - {postfix}]",
#             colour="green",
#             leave=False,
#     ):
#         time.sleep(max_execution_time / 1000)
#
#         if not p.is_alive():
#             p.join()
#             result = recv_end.recv()
#             break
#
#     if p.is_alive():
#         p.terminate()
#         p.join()
#         raise TimeoutException(
#             "Failed to complete process within time limit.",
#             func=function,
#             max_time=max_execution_time,
#         )
#     else:
#         p.join()
#         result = recv_end.recv()
#
#     if isinstance(result, Exception):
#         raise result
#     else:
#         return result
#
#
# def truncate_spline(f, r_t, a):
#     r"""
#     Takes the function ``f`` and returns a truncated equivalent of it, which becomes
#     .. math::
#     f'(x) = f(r_t) \left(\frac{x}{r_t}\right)**(r_t*df/dx(r_t)/f(r_t))
#     This preserves the slope and continuity of the function be yields a monotonic power law at large :math:`r`.
#     Parameters
#     ----------
#     f: InterpolatedUnivariateSpline
#         The function to truncate
#     r_t: float
#         The scale radius
#     a: float
#         Truncation rate. Higher values cause transition more quickly about :math:`r_t`.
#     Returns
#     -------
#     callable
#         The new function.
#     """
#     _gamma = r_t * f(r_t, 1) / f(r_t)  # This is the slope.
#     return lambda x, g=_gamma, _a=a, r=r_t: f(x) * _truncator_function(_a, r, x) + (
#             1 - _truncator_function(_a, r, x)
#     ) * (f(r) * _truncator_function(-g, r, x))
#
#
# def integrate_mass(profile, rr):
#     mass_int = lambda r: profile(r) * r * r
#     mass = np.zeros(rr.shape)
#     for i, r in enumerate(rr):
#         mass[i] = 4.0 * np.pi * quad(mass_int, 0, r)[0]
#     return mass
#
#
# def integrate(profile, rr):
#     ret = np.zeros(rr.shape)
#     rmax = rr[-1]
#     for i, r in enumerate(rr):
#         ret[i] = quad(profile, r, rmax)[0]
#     return ret
#
#
# def integrate_toinf(profile, rr):
#     ret = np.zeros(rr.shape)
#     rmax = rr[-1]
#     for i, r in enumerate(rr):
#         ret[i] = quad(profile, r, rmax)[0]
#     ret[:] += quad(profile, rmax, np.inf, limit=100)[0]
#     return ret
#
#
# def generate_particle_radii(r, m, num_particles, r_max=None, prng=None):
#     prng = parse_prng(prng)
#     if r_max is None:
#         ridx = r.size
#     else:
#         ridx = np.searchsorted(r, r_max)
#     mtot = m[ridx - 1]
#     u = prng.uniform(size=num_particles)
#     P_r = np.insert(m[:ridx], 0, 0.0)
#     P_r /= P_r[-1]
#     r = np.insert(r[:ridx], 0, 0.0)
#     radius = np.interp(u, P_r, r, left=0.0, right=1.0)
#     return radius, mtot
#
#
# field_label_map = {
#     "density": "$\\rho_g$ (g cm$^{-3}$)",
#     "temperature": "kT (keV)",
#     "pressure": "P (erg cm$^{-3}$)",
#     "entropy": "S (keV cm$^{2}$)",
#     "dark_matter_density": "$\\rho_{\\rm DM}$ (g cm$^{-3}$)",
#     "electron_number_density": "n$_e$ (cm$^{-3}$)",
#     "stellar_mass": "M$_*$ (M$_\\odot$)",
#     "stellar_density": "$\\rho_*$ (g cm$^{-3}$)",
#     "dark_matter_mass": "$M_{\\rm DM}$ (M$_\\odot$)",
#     "gas_mass": "M$_g$ (M$_\\odot$)",
#     "total_mass": "M$_{\\rm tot}$ (M$_\\odot$)",
#     "gas_fraction": "f$_{\\rm gas}$",
#     "magnetic_field_strength": "B (G)",
#     "gravitational_potential": "$\\Phi$ (kpc$^2$ Myr$^{-2}$)",
#     "gravitational_field": "g (kpc Myr$^{-2}$)",
# }
#
#
# class HDF5FileHandler:
#     """
#     A handler for managing scratch space within an HDF5 file or group, supporting
#     dynamic creation, resizing, and maintenance of temporary datasets and groups.
#     """
#
#     def __init__(self, handle: Union[h5py.File, h5py.Group, str], mode="r+"):
#         """
#         Initialize the scratch space handler with an HDF5 file or group handle.
#
#         Parameters
#         ----------
#         handle : str or h5py.File or h5py.Group
#             The path to the HDF5 file, or an open HDF5 file/group handle.
#         mode : str, optional
#             The mode to open the file in ('r+' for read/write, 'a' for append), by default 'r+'.
#         """
#         if isinstance(handle, str):
#             self.handle = h5py.File(handle, mode=mode)
#         elif isinstance(handle, (h5py.File, h5py.Group)):
#             self.handle = handle
#         else:
#             raise TypeError("Handle must be a file path, an instance of h5py.File, or h5py.Group")
#         self.filename = handle if isinstance(handle, str) else handle.file.filename
#         self.mode = mode
#
#     def __getitem__(self, key):
#         """
#         Access a dataset or group by key.
#         """
#         return self.handle[key]
#
#     def __delitem__(self, key):
#         del self.handle[key]
#
#     def __contains__(self, item):
#         """
#         Check if a dataset or group exists in the file/group.
#         """
#         return item in self.handle
#
#     def __len__(self):
#         """
#         Return the number of items in the file/group.
#         """
#         return len(self.handle)
#
#     @property
#     def attrs(self):
#         """
#         Return the attributes of the HDF5 file or group.
#         """
#         return self.handle.attrs
#
#     def get(self, item, default=None):
#         """
#         Retrieve an item by key or return a default if not found.
#         """
#         return self.handle.get(item, default)
#
#     def switch_mode(self, mode: str):
#         """
#         Switch the HDF5 file's mode dynamically. Only applicable if the handler was initialized with a file path.
#         """
#         if isinstance(self.handle, h5py.File):
#             self.handle.close()
#             self.handle = h5py.File(self.filename, mode=mode)
#         else:
#             raise ValueError("Cannot switch mode when initialized with an HDF5 group handle.")
#
#     def keys(self):
#         """
#         Return a list of dataset and group keys in the file/group.
#         """
#         return list(self.handle.keys())
#
#     def items(self):
#         """
#         Return a list of dataset and group items in the file/group.
#         """
#         return list(self.handle.items())
#
#     def write_data(self, dataset_name, data):
#         """
#         Write or overwrite a dataset in the file/group.
#
#         Parameters
#         ----------
#         dataset_name : str
#             The name of the dataset to write.
#         data : array-like
#             The data to write to the dataset.
#         """
#         if dataset_name in self.handle:
#             del self.handle[dataset_name]  # Overwrite the existing dataset
#         self.handle.create_dataset(dataset_name, data=data)
#
#     def flush(self):
#         """
#         Flush changes to disk without closing the file/group.
#         """
#         if self.handle is not None:
#             self.handle.flush()
#
#     def create_group(self, *args, **kwargs):
#         """
#         Create a group within the file/group.
#         """
#         return self.handle.create_group(*args, **kwargs)
#
#     def close(self):
#         """
#         Close the HDF5 file if it's a top-level file handle.
#         """
#         if isinstance(self.handle, h5py.File):
#             self.handle.close()
#
#     def __enter__(self):
#         """
#         Enter runtime context for safe handling.
#         """
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         """
#         Exit runtime context, ensuring the file is closed if applicable.
#         """
#         self.close()
#
#     def __str__(self):
#         return f"HDF5FileHandler for '{self.handle.name}' with {len(self)} items"
#
#     def __repr__(self):
#         return f"<HDF5FileHandler(handle='{self.handle.name}', num_items={len(self)})>"
