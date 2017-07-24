from collections import OrderedDict
from six import add_metaclass
from yt import savetxt, mylog, YTArray
import h5py
import os

equilibrium_model_registry = {}

class RegisteredClusterModel(type):
    def __init__(cls, name, b, d):
        type.__init__(cls, name, b, d)
        if hasattr(cls, "_type_name"):
            equilibrium_model_registry[cls._type_name] = cls

@add_metaclass(RegisteredClusterModel)
class ClusterModel(object):

    def __init__(self, num_elements, fields):
        self.num_elements = num_elements
        self.fields = fields

    @classmethod
    def from_h5_file(cls, filename):
        r"""
        Generate an equilibrium model from an HDF5 file. 

        Parameters
        ----------
        filename : string
            The name of the file to read the model from.

        Examples
        --------
        >>> from cluster_generator import ClusterModel
        >>> hse_model = ClusterModel.from_h5_file("hse_model.h5")
        """
        f = h5py.File(filename)

        model_type = f["model_type"].value
        num_elements = f["num_elements"].value
        fnames = list(f['fields'].keys())

        f.close()

        fields = OrderedDict()
        for field in fnames:
            fields[field] = YTArray.from_hdf5(filename, dataset_name=field,
                                              group_name="fields").in_base("galactic")

        return equilibrium_model_registry[model_type](num_elements, fields)

    def __getitem__(self, key):
        return self.fields[key]

    def keys(self):
        return self.fields.keys()

    def write_model_to_ascii(self, output_filename, in_cgs=False, clobber=False):
        """
        Write the equilibrium model to an HDF5 file.

        Parameters
        ----------
        output_filename : string
            The file to write the model to.
        in_cgs : boolean, optional
            Whether to convert the units to cgs before writing. Default False.
        clobber : boolean, optional
            Overwrite an existing file with the same name. Default False.
        """
        if os.path.exists(output_filename) and not clobber:
            raise IOError("Cannot create %s. It exists and clobber=False." % output_filename)
        field_list = list(self.fields.keys())
        num_fields = len(field_list)
        name_fmt_str = " Fields\n"+" %s\t"*(num_fields-1)+"%s"
        header = name_fmt_str % tuple(field_list)

        if in_cgs:
            fields = OrderedDict()
            for k, v in self.fields.items():
                fields[k] = v.in_cgs()
        else:
            fields = self.fields

        if "particle_mass" in fields:
            particle_mass = fields.pop("particle_mass")
            header += "\n Particle mass = %g %s" % (particle_mass[0].v, 
                                                    particle_mass[0].units)

        savetxt(output_filename, list(fields.values()), header=header)

    def write_model_to_h5(self, output_filename, in_cgs=False, clobber=False):
        """
        Write the equilibrium model to an HDF5 file.

        Parameters
        ----------
        output_filename : string
            The file to write the model to.
        in_cgs : boolean, optional
            Whether to convert the units to cgs before writing. Default False.
        clobber : boolean, optional
            Overwrite an existing file with the same name. Default False.
        """
        if os.path.exists(output_filename) and not clobber:
            raise IOError("Cannot create %s. It exists and clobber=False." % output_filename)
        f = h5py.File(output_filename, "w")
        f.create_dataset("model_type", data=self._type_name)
        f.create_dataset("num_elements", data=self.num_elements)
        f.close()
        for field in list(self.fields.keys()):
            if in_cgs:
                if field == "temperature":
                    fd = self.fields[field].to_equivalent("K", "thermal")
                else:
                    fd = self.fields[field]
                fd.in_cgs().write_hdf5(output_filename, dataset_name=field, 
                                       group_name="fields")
            else:
                self.fields[field].write_hdf5(output_filename, dataset_name=field,
                                              group_name="fields")

    def set_field(self, name, value):
        """
        Set a field with name *name* to value *value*, which is a YTArray.
        The array will be checked to make sure that it has the appropriate size.
        """
        if not isinstance(value, YTArray):
            raise TypeError("value needs to be a YTArray")
        if len(value) == self.num_elements:
            if name in self.fields:
                mylog.warning("Overwriting field %s." % name)
            self.fields[name] = value
        else:
            raise ValueError("The length of the array needs to be %d elements!"
                             % self.num_elements)

