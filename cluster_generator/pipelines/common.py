from typing import Type, Optional, TYPE_CHECKING

import unyt

from cluster_generator.pipelines.abc import Pipeline
from cluster_generator.pipelines.scratch_handlers.radial import RadialScratchHandler
from cluster_generator.pipelines.solvers import Solver
from cluster_generator.utilities.math import integrate
from cluster_generator.pipelines.solvers.common import solver
from cluster_generator.utilities.physics import mu,mp
from cluster_generator.pipelines.conditions.common import ALWAYS,NEVER
from scipy.interpolate import InterpolatedUnivariateSpline
if TYPE_CHECKING:
    from cluster_generator.pipelines.abc import Pipeline
    from cluster_generator.grids.grids import Grid

class DensityTemperaturePipeline(Pipeline):
    DEFAULT_SCRATCH_CLASS: Type[RadialScratchHandler] = RadialScratchHandler

    @property
    def scratch(self) -> RadialScratchHandler:
        return super().scratch

    # noinspection PyMethodParameters
    @solver
    def setup_radial_scratch(pipeline: 'DensityTemperaturePipeline', grid: 'Grid'):
        # For the grid, ensure that there is a level corresponding to
        # the correct grid level.
        if grid.level.index not in pipeline.scratch.levels:
            level_scratch = pipeline.scratch.add_level(grid.level.index)
        else:
            level_scratch = pipeline.scratch.get_level(grid.level.index)
        # For each of the relevant profiles, we want to add a blank array
        # to the profile to ensure that everything is ready to go.
        for profile in ['pressure','d_phi','phi']:
            level_scratch.add_profile(profile)

        # Ensure that this grid's radii are in the scratch space.
        level_scratch.add_radii_from_grid(grid,buffer=3)

    # noinspection PyMethodParameters
    @setup_radial_scratch.mark_pipeline
    def _sp_setup_radial_scratch(pipeline: 'DensityTemperaturePipeline'):
        # Setup the radial scratch space in the pipeline. All this does is
        # ensure that the pipeline scratch actually exists before runtime.
        if pipeline._scratch is None:
            pipeline.setup_scratch()

    @solver
    def compute_pressure(pipeline,grid: "Grid"):
        level_scratch = pipeline.scratch.get_level(grid.level.index)

        radii, slc = level_scratch.get_radii_and_slc_from_grid(grid)
        temp, dens = unyt.unyt_array(pipeline.model.temperature_profile(radii),'keV'),unyt.unyt_array(pipeline.model.density_profile(radii),'Msun/kpc**3')
        pressure = unyt.unyt_array((temp*dens)/(mp*mu)).to_value("Msun/(kpc*Myr**2)")

        level_scratch.profiles['pressure'][slc] = pressure

        # compute the spline
        pressure_spline = InterpolatedUnivariateSpline(radii,pressure)

        grid.add_field_from_function(pressure_spline,'pressure',dtype='float64',units="Msun/(kpc*Myr**2)",geometry=pipeline.model.geometry)

        pipeline._context['pressure_spline'] = pressure_spline

    @solver
    def compute_potential(pipeline,grid: 'Grid'):
        pressure_spline = pipeline._context['pressure_spline']
        dens = pipeline.model.density_profile
        level_scratch = pipeline.scratch.get_level(grid.level.index)

        radii, slc = level_scratch.get_radii_and_slc_from_grid(grid)
        integrand = lambda x: pressure_spline(x,1)/dens(x)


        if grid.level.index == 0:
            # This is a base grid, we utilize a zero-point boundary condition
            boundary = 0
        else:
            # We need to interpolate from the level above.
            parent_radii_slc = pipeline.scratch.get_level(grid.level.index-1).get_radii_slice(radii[0],radii[-1])
            boundary = pipeline.scratch.get_level(grid.level.index-1).profiles['phi'][parent_radii_slc][-1]

        # performing the integration step
        potential = integrate(integrand,radii) + boundary
        level_scratch.profiles['phi'][slc] = potential

        p_spline = level_scratch.get_spline_from_grid('phi',grid)
        grid.add_field_from_function(p_spline,'potential',dtype='float64',units='kpc**2/Myr**2',geometry=pipeline.model.geometry)

        # Compute the total density
        #_interpolation_kernel = lambda x: (x**2)*integrand(x)
        #_interpolation_spline = InterpolatedUnivariateSpline(radii,_interpolation_kernel(radii))
#
        #laplacian = lambda x: (1/x**2)*(_interpolation_spline(x,1))
        #lame_r = pipeline.model.geometry.lame_functions[0]
#
        #grid_coordinates = grid.get_coordinates()
        #converter = pipeline.model.geometry.build_converter(pipeline.model.grid_manager.AXES)
        ##geo_coordinates = pipeline.model.geometry.from_cartesian(*converter(grid_coordinates),np.zeros_like(gr))
#
        #grid.add_field('total_density',units='Msun/kpc**3',data=(1/lame_r(*geo_coordinates)**2)*laplacian(geo_coordinates[0]))


    _BASE_PROCEDURE = {
        'start': [('setup_radial_scratch',ALWAYS)],
        'setup_radial_scratch':[('compute_pressure',ALWAYS)],
        'compute_pressure':[('compute_potential',ALWAYS)],
        'compute_potential': [('end',ALWAYS)]
    }

    def __init__(self,*args,**kwargs):
        kwargs['procedure'] = self.__class__._BASE_PROCEDURE

        super().__init__(*args,**kwargs)


if __name__ == '__main__':
    DensityTemperaturePipeline.logger.setLevel('TRACE')

    import h5py

    with h5py.File("random_test.hdf5",'w') as fio:
        p = DensityTemperaturePipeline()
        p.to_hdf5(fio)

    with h5py.File('random_test.hdf5','r+') as fio:
        p = DensityTemperaturePipeline.from_hdf5(fio)
        p.ensure_setup()
        p.ensure_setup()
    #print(DensityTemperaturePipeline.solver_registry)
    #for k,v in DensityTemperaturePipeline.__dict__.items():
    #    print(k,v)
#
    #print(DensityTemperaturePipeline.solver_registry['setup_radial_scratch']._setup_in_pipeline(p))