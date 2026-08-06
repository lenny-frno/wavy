"""
Module for quicklook fct
"""

# imports
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from abc import abstractmethod
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import cmocean
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import logging

logger = logging.getLogger(__name__)

# own imports
from wavy.wconfig import load_or_default
from wavy.utils import parse_date
from wavy.utils import compute_quantiles
from wavy.validationmod import linreg_evm, linreg_std

# read yaml config files:
region_dict = load_or_default("region_cfg.yaml")
variable_info = load_or_default("variable_def.yaml")
model_dict = load_or_default("model_cfg.yaml")
quicklook_dict = load_or_default("quicklook_cfg.yaml")
region_dict = load_or_default("region_cfg.yaml")
variable_info = load_or_default("variable_def.yaml")
model_dict = load_or_default("model_cfg.yaml")
quicklook_dict = load_or_default("quicklook_cfg.yaml")


class quicklook_class_sat:

    def _check_varalias(self, **kwargs):
        """
        Check if the varalias is a list or a string. If it is a list, get the varalias from kwargs or use the first element of the list. If it is a string, use it as the varalias. Return the varalias and the corresponding units to plot.
        """
        if isinstance(self.varalias, list):
            varalias = kwargs.get("varalias", self.varalias[0])
            assert varalias in self.varalias, "varalias must be one of {}".format(
                self.varalias
            )
            assert isinstance(varalias, str), "varalias argument should be a string"
            idx_units = np.argwhere(np.array(self.varalias) == varalias)[0][0]
            units_to_plot = self.units[idx_units]
        else:
            varalias = self.varalias
            units_to_plot = self.units

        return varalias, units_to_plot

    def _set_plot_var(self, **kwargs):
        """
        Set the variable to plot, the longitude and latitude arrays, and other plotting parameters.
        """
        varalias, units_to_plot = self._check_varalias(**kwargs)
        try:
            plot_var = self.vars[varalias]
            plot_lons = self.vars.lons
            plot_lats = self.vars.lats
            plot_var_obs = None
            plot_var_model = None
        except Exception as e:
            list_vars = list(self.vars.variables)
            assert "model_" + varalias in list_vars, (
                "model_{}".format(varalias)
                + " is missing in "
                + "the dataset, if you would like to "
                + "validate another variable, please "
                + "specify with varalias."
            )
            assert "obs_" + varalias in list_vars, (
                "obs_{}".format(varalias)
                + " is missing in "
                + "the dataset, if you would like to "
                + "validate another variable, please "
                + "specify with varalias."
            )
            plot_var = self.vars["obs_" + varalias]
            plot_lons = self.vars.obs_lons
            plot_lats = self.vars.obs_lats
            plot_var_obs = self.vars["obs_" + varalias]
            plot_var_model = self.vars["model_" + varalias]

        if str(type(self)) == "<class 'wavy.model_module.model_class'>":
            if len(plot_lons.shape) < 2:
                plot_lons, plot_lats = np.meshgrid(plot_lons, plot_lats)

        fs = kwargs.get("fs", 12)

        vartype = variable_info[varalias].get("type", "default")
        if kwargs.get("cmap") is None:
            if vartype == "cyclic":
                cmap = mpl.cm.twilight
            else:
                cmap = cmocean.cm.amp
        else:
            cmap = kwargs.get("cmap")

        return plot_var, plot_lons, plot_lats, plot_var_obs, plot_var_model, fs, cmap

    def plot_sat(self, **kwargs):
        """
        Placeholder function for plotting satellite data.
        """
        if kwargs.get(
            "plot_xtrack_pulse_limited_fpr"
        ):  # check if it works when plot_xtrack_pulse_limited_fpr is not None
            domain = kwargs.get("domain", "lonlat")
            number_of_seeds = kwargs.get("number_of_seeds", 100)
            lons_perp, lats_perp, _, _, ls_idx_lst = self._generate_xtrack_footprints(
                domain=domain, number_of_seeds=number_of_seeds
            )
            sc2 = ax.scatter(
                lons_perp,
                lats_perp,
                s=0.2,
                c="b",
                marker=".",
                edgecolor="face",
                transform=ccrs.PlateCarree(),
            )

    def plot_map(
        self,
        projection,
        plot_lons,
        plot_lats,
        plot_var,
        varalias,
        units_to_plot,
        cmap,
        fs,
        **kwargs,
    ):
        """
        Plot a map with the given projection and keyword arguments.

        Keyword arguments:
        vmin -- minimum value for the color scale (default: 0)
        vmax -- maximum value for the color scale (default: np.nanmax(plot_var))
        levels_incr -- increment for contour levels (default: 0.5)
        levels -- contour levels (default: np.arange(vmin, vmax, levels_incr))
        cflevels -- filled contour levels (default: levels)
        clevels -- contour line levels (default: levels)
        zorder_land -- z-order for land feature (default: 10)
        land_mask_resolution -- resolution for land mask (default: "i")
        lonmax, lonmin, latmax, latmin -- map boundaries (default: based on plot_lons and plot_lats)
        poi -- point of interest to plot (default: None)
        fs -- font size (default: 12)
        cmap -- colormap (default: None)
        """

        vmin = kwargs.get("vmin", 0)
        vmax = kwargs.get("vmax", np.nanmax(plot_var))

        levels_incr = kwargs.get("levels_incr", 0.5)

        levels = kwargs.get("levels", np.arange(vmin, vmax, levels_incr))

        cflevels = kwargs.get("cflevels", levels)
        clevels = kwargs.get("clevels", levels)

        norm = mpl.colors.BoundaryNorm(levels, cmap.N)

        zorder_land = kwargs.get("zorder_land", 10)

        land = cfeature.GSHHSFeature(
            scale=kwargs.get("land_mask_resolution", "i"),
            levels=[1],
            facecolor=cfeature.COLORS["land"],
        )
        projection = _check_projection(projection)

        lonmax, lonmin = np.max(plot_lons), np.min(plot_lons)
        latmax, latmin = np.max(plot_lats), np.min(plot_lats)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=projection)

        # add land
        ax.add_geometries(
            land.intersecting_geometries([-180, 180, 0, 90]),
            crs=ccrs.PlateCarree(),  # Be careful with the transform, it should be the same as the data here (lon,lat)
            facecolor=cfeature.COLORS["land"],
            edgecolor="black",
            linewidth=1,
            zorder=zorder_land,
        )

        # add sea map
        # ax.add_wmts("https://cache.kartverket.no/v1/wmts", 'sjokartraster')

        # - add land color
        ax.add_feature(land, facecolor="burlywood", alpha=0.5)

        lonmax, lonmin, latmax, latmin = _set_lonlat_minmax(
            plot_lons, plot_lats, **kwargs
        )

        if poi := kwargs.get("poi"):  # check if it works when poi is None
            lonmax, lonmin, latmax, latmin = plot_poi(
                ax, poi, plot_lons, plot_lats, **kwargs
            )
        # plot sat
        self.plot_sat(**kwargs)

        sc, c = plot_var_field(
            ax,
            plot_var,
            plot_lons,
            plot_lats,
            vmin,
            vmax,
            cmap,
            cflevels,
            clevels,
            norm,
            **kwargs,
        )

        # axes for colorbar
        axins = inset_axes(
            ax,
            width="5%",  # width = 5% of parent_bbox width
            height="100%",  # height : 50%
            loc="lower left",
            bbox_to_anchor=(1.01, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        # - colorbar
        if kwargs.get("cbar", True) is True:
            cbar = fig.colorbar(
                sc,
                cax=axins,
                label=varalias + " [" + units_to_plot + "]",
                ticks=levels,
            )
            cbar.ax.set_ylabel(units_to_plot, size=fs)
            cbar.ax.tick_params(labelsize=fs)

        # - add extent
        _set_extent(ax, lonmax, lonmin, latmax, latmin, projection, **kwargs)

        # ax.coastlines(color='k')
        if projection == ccrs.PlateCarree():
            gl = ax.gridlines(
                draw_labels=True,
                crs=projection,
                linewidth=1,
                color="grey",
                alpha=0.4,
                linestyle="-",
            )
            gl.top_labels = False
            gl.right_labels = False

        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

        auto_title = (
            self.nID
            + "\n"
            + "from "
            + (parse_date(str(self.vars["time"][0].values))).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            + " to "
            + (parse_date(str(self.vars["time"][-1].values))).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
        title = kwargs.get("title", auto_title)
        ax.set_title(title)
        # plot from quicklook config file
        if (
            "region" in vars(self).keys()
            and self.region in quicklook_dict
            and "poi" in quicklook_dict[self.region]
        ):
            for poi in quicklook_dict[self.region]["poi"]:
                pname = quicklook_dict[self.region]["poi"][poi]["name"]
                plat = quicklook_dict[self.region]["poi"][poi]["lat"]
                plon = quicklook_dict[self.region]["poi"][poi]["lon"]
                scp = ax.scatter(
                    plon,
                    plat,
                    s=20,
                    c=quicklook_dict[self.region]["poi"][poi].get("color", "b"),
                    marker=quicklook_dict[self.region]["poi"][poi]["marker"],
                    transform=ccrs.PlateCarree(),
                )  # switch from projection
            ax.text(plon, plat, pname, transform=ccrs.PlateCarree(), zorder=100)
        # fig.suptitle('', fontsize=16) # unused
        if kwargs.get("show", True) is True:
            plt.show()
        return fig, ax

    def plot_timeseries(
        self, mode, plot_var, plot_var_model, varalias, units_to_plot, **kwargs
    ):
        """
        Plot a time series of the specified variable.
        """
        if mode == "comb":
            fig = plt.figure(figsize=(9, 3.5))
            ax = fig.add_subplot(111)
            colors = ["k", "r"]
            ax.plot(
                self.vars["time"],
                plot_var,
                color=colors[0],
                linestyle=kwargs.get("linestyle", ""),
                label=self.nID,
                marker="o",
                alpha=0.5,
                ms=2,
            )
            colors = ["k", "r"]
            ax.plot(
                self.vars["time"],
                plot_var,
                color=colors[0],
                linestyle=kwargs.get("linestyle", ""),
                label=self.nID,
                marker="o",
                alpha=0.5,
                ms=2,
            )
            try:
                if "model" in vars(self):
                    label_scdplot = self.model
                else:
                    label_scdplot = self.nID
                ax.plot(
                    self.vars["time"],
                    plot_var_model,
                    color=colors[1],
                    linestyle=kwargs.get("linestyle", ""),
                    label=label_scdplot,
                    marker="o",
                    alpha=0.5,
                    ms=2,
                )
            except Exception as e:
                pass
            plt.ylabel(varalias + " [" + units_to_plot + "]")
            plt.legend(loc="best")
            plt.tight_layout()
            # ax.set_title()
            if kwargs.get("show", True) is True:
                plt.show()
            return fig, ax

        elif mode == "indiv":
            fig = plt.figure(figsize=(9, 3.5))
            ax = fig.add_subplot(111)
            for oco in self.ocos:
                label = oco.name

                ax.plot(
                    oco.vars["time"],
                    oco.vars[varalias],
                    linestyle=kwargs.get("linestyle", ""),
                    label=label,
                    marker="o",
                    alpha=0.5,
                    ms=2,
                )
            try:
                label = self.model
                ax.plot(
                    self.vars["time"],
                    plot_var_model,
                    color=colors[1],
                    linestyle=kwargs.get("linestyle", ""),
                    label=label,
                    marker="o",
                    alpha=0.5,
                    ms=2,
                )
            except Exception as e:
                pass
            plt.ylabel(varalias + " [" + units_to_plot + "]")
            plt.legend(loc="best")
            plt.tight_layout()
            # ax.set_title()
            if kwargs.get("show", True) is True:
                plt.show()
            return fig, ax
        else:
            raise ValueError(
                "mode must be either 'comb' or 'indiv', got {}".format(mode)
            )

    def plot_scat(
        self, plot_var_obs, plot_var_model, varalias, units_to_plot, **kwargs
    ):
        lq = np.arange(0.01, 1.01, 0.01)
        lq = kwargs.get("lq", lq)
        modq = compute_quantiles(plot_var_model, lq)
        obsq = compute_quantiles(plot_var_obs, lq)

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        colors = ["k"]

        ax.plot(
            plot_var_obs,
            plot_var_model,
            linestyle="None",
            color=colors[0],
            marker="o",
            alpha=0.5,
            ms=2,
            label="data",
        )

        # add quantiles
        ax.plot(obsq, modq, "r", label="QQ")

        # 45 degree line for orientation
        ax.axline((0, 0), (1, 1), lw=0.5, color="grey", ls="--", label="45 deg")

        # linreg_evm line
        if kwargs.get("evm_regression_line") is True:
            rl = linreg_evm(plot_var_obs, plot_var_model, **kwargs)
            self.EVMreg = dict({"intercept": rl[1], "slope": rl[0]})
            ax.axline(
                xy1=(0, rl[1]),
                slope=rl[0],
                color=kwargs.get("evm_regression_col", "lightblue"),
                lw=kwargs.get("evm_regression_lw", 1),
                ls=kwargs.get("evm_regression_ls", "-"),
                label="EVM-regr",
            )

        # std linreg line
        if kwargs.get("std_regression_line") is True:
            rl = linreg_std(plot_var_obs, plot_var_model, **kwargs)
            self.linreg = dict({"intercept": rl["intercept"], "slope": rl["slope"]})
            ax.axline(
                xy1=(0, rl["intercept"]),
                slope=rl["slope"],
                color=kwargs.get("std_regression_col", "lightblue"),
                lw=kwargs.get("std_regression_lw", 1),
                ls=kwargs.get("std_regression_ls", "-"),
                label="linregr",
            )

        # add axis labels
        plt.xlabel("obs (" + self.nID + ")")
        plt.ylabel("models (" + self.model + ")")

        maxv = np.nanmax([self.vars["model_" + varalias], self.vars["obs_" + varalias]])
        minv = 0
        plt.xlim([minv, maxv * 1.05])
        plt.ylim([minv, maxv * 1.05])

        ax.set_title(varalias + "[" + units_to_plot + "]")
        plt.legend()

        plt.tight_layout()

        # ax.set_title()
        if kwargs.get("show", True) is True:
            plt.show()
        return fig, ax

    def plot_hist(
        self, plot_var_obs, plot_var_model, varalias, units_to_plot, fs, **kwargs
    ):
        lq = np.arange(0.01, 1.01, 0.01)
        lq = kwargs.get("lq", lq)
        modq = compute_quantiles(plot_var_model, lq)
        obsq = compute_quantiles(plot_var_obs, lq)

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)

        # 2d histogram
        lmin = 0
        lmax = np.nanmax([plot_var_obs, plot_var_model]) * 1.05
        plt.hist2d(
            plot_var_obs,
            plot_var_model,
            bins=kwargs.get("bins", 100),
            range=[[lmin, lmax], [lmin, lmax]],
            norm=kwargs.get("norm", mpl.colors.LogNorm()),
            cmap=kwargs.get("cmap", mpl.cm.gray),
            cmin=kwargs.get("cmin", 1),
        )
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label("Frequency", size=fs)
        plt.xlim([lmin, lmax])
        plt.ylim([lmin, lmax])

        # add quantiles
        ax.plot(obsq, modq, "r", label="QQ")

        # 45 degree line for orientation
        ax.axline((0, 0), (1, 1), lw=0.5, color="grey", ls="--", label="45 deg")

        # linreg_evm line
        if kwargs.get("evm_regression_line") is True:
            rl = linreg_evm(plot_var_obs, plot_var_model, **kwargs)
            self.EVMreg = dict({"intercept": rl[1], "slope": rl[0]})
            ax.axline(
                xy1=(0, rl[1]),
                slope=rl[0],
                color=kwargs.get("evm_regression_col", "lightblue"),
                lw=kwargs.get("evm_regression_lw", 1),
                ls=kwargs.get("evm_regression_ls", "-"),
                label="EVM-regr",
            )

        # std linreg line
        if kwargs.get("std_regression_line") is True:
            rl = linreg_std(plot_var_obs, plot_var_model, **kwargs)
            self.linreg = dict({"intercept": rl["intercept"], "slope": rl["slope"]})
            ax.axline(
                xy1=(0, rl["intercept"]),
                slope=rl["slope"],
                color=kwargs.get("std_regression_col", "lightblue"),
                lw=kwargs.get("std_regression_lw", 1),
                ls=kwargs.get("std_regression_ls", "-"),
                label="linregr",
            )

        # add axis labels
        plt.xlabel("obs (" + self.nID + ")")
        plt.ylabel("models (" + self.model + ")")

        ax.set_title(varalias + "[" + units_to_plot + "]")
        plt.legend()

        plt.tight_layout()

        if kwargs.get("show", True) is True:
            plt.show()
        return fig, ax

    def quicklook(self, a=False, projection=None, **kwargs):
        """
        Enables to explore the class object (and retrieved results)
        by plotting time series and map.

        param:
            m - map figure (True/False)
            ms - map figure + scatter (True/False)
            ts - time series (True/False)
            a - all figures (True/False)
            projection - specified projection for cartopy

        return:
            figures
        """
        # TODO refactor different plotting functions into separate functions for clarity and maintainability

        import matplotlib as mpl
        import cmocean

        log_level = str(kwargs.get("logging", "WARNING").upper())
        logger.setLevel(getattr(logging, log_level, logging.WARNING))
        # settings
        m = kwargs.get("m", a)
        ts = kwargs.get("ts", a)
        scat = kwargs.get("sc", False)
        hst = kwargs.get("hist", False)
        mode = kwargs.get("mode", "comb")  # comb, indiv
        logger.debug(
            f"quicklook kwargs: m={m}, ts={ts}, scat={scat}, hst={hst}, mode={mode}"
        )

        varalias, units_to_plot = self._check_varalias(**kwargs)
        plot_var, plot_lons, plot_lats, plot_var_obs, plot_var_model, fs, cmap = (
            self._set_plot_var(**kwargs)
        )

        if m is True:
            fig, ax = self.plot_map(
                projection,
                plot_lons,
                plot_lats,
                plot_var,
                varalias,
                units_to_plot,
                cmap,
                fs,
                **kwargs,
            )

        if ts is True:
            fig, ax = self.plot_timeseries(
                mode, plot_var, plot_var_model, varalias, units_to_plot, **kwargs
            )

        if scat is True:
            fig, ax = self.plot_scat(
                plot_var_obs, plot_var_model, varalias, units_to_plot, **kwargs
            )

        if hst is True:
            fig, ax = self.plot_hist(
                plot_var_obs, plot_var_model, varalias, units_to_plot, fs, **kwargs
            )
        if kwargs.get("show") is False:
            return fig, ax

    def quick_anim():
        pass


def _check_projection(projection):
    """
    Checks if a projection is specified. If not, defaults to PlateCarree projection."""
    if projection is None:
        projection = ccrs.PlateCarree()
        logger.debug("projection not specified, using default: PlateCarree")
    logger.info(f"projection: {projection}")
    return projection


def _set_lonlat_minmax(lons, lats, **kwargs):
    """
    Sets the min and max values for longitude and latitude based on provided keyword arguments or defaults to the min/max of the provided arrays.
    """
    if kwargs.get("lonmax") is not None:
        lonmax = kwargs.get("lonmax")
    else:
        lonmax = np.max(lons)
    if kwargs.get("latmax") is not None:
        latmax = kwargs.get("latmax")
    else:
        latmax = np.max(lats)
    if kwargs.get("lonmin") is not None:
        lonmin = kwargs.get("lonmin")
    else:
        lonmin = np.min(lons)
    if kwargs.get("latmin") is not None:
        latmin = kwargs.get("latmin")
    else:
        latmin = np.min(lats)
    return lonmax, lonmin, latmax, latmin


def _set_polar_extent(ax, lonmax, lonmin, latmax, latmin, projection, **kwargs):
    """
    Sets the extent of the map for polar stereographic projections based on the provided longitude and latitude ranges and keyword arguments.
    """
    map_extent_multiplicator = kwargs.get("map_extent_multiplicator", 0.1)
    map_extent_multiplicator_lon = kwargs.get(
        "map_extent_multiplicator_lon", map_extent_multiplicator
    )
    map_extent_multiplicator_lat = kwargs.get(
        "map_extent_multiplicator_lat", map_extent_multiplicator
    )
    lon_range = lonmax - lonmin
    lat_range = latmax - latmin
    lat0 = projection._proj4_params.get("lat_0")

    if lat0 > 60:
        # Northern hemisphere polar view
        extent = [
            max(-180, lonmin - lon_range * map_extent_multiplicator_lon),
            min(180, lonmax + lon_range * map_extent_multiplicator_lon),
            max(30, latmin - lat_range * map_extent_multiplicator_lat),
            90,
        ]
        logger.debug("Northern hemisphere projection detected")

    elif lat0 < -60:
        # Southern hemisphere polar view
        extent = [
            max(-180, lonmin - lon_range * map_extent_multiplicator_lon),
            min(180, lonmax + lon_range * map_extent_multiplicator_lon),
            -90,
            min(-30, latmax + lat_range * map_extent_multiplicator_lat),
        ]
        logger.debug("Southern hemisphere projection detected")
    else:
        extent = [
            lonmin - lon_range * map_extent_multiplicator_lon,
            lonmax + lon_range * map_extent_multiplicator_lon,
            latmin - lat_range * map_extent_multiplicator_lat,
            latmax + lat_range * map_extent_multiplicator_lat,
        ]
    return extent


def _set_extent(ax, lonmax, lonmin, latmax, latmin, projection, **kwargs):
    if kwargs.get("map_extent_llon") is None:

        map_extent_multiplicator = kwargs.get("map_extent_multiplicator", 0.1)
        map_extent_multiplicator_lon = kwargs.get(
            "map_extent_multiplicator_lon", map_extent_multiplicator
        )
        map_extent_multiplicator_lat = kwargs.get(
            "map_extent_multiplicator_lat", map_extent_multiplicator
        )

        logger.debug(
            f"map_extent_multiplicator_lon: {map_extent_multiplicator_lon}, map_extent_multiplicator_lat: {map_extent_multiplicator_lat}"
        )

        lon_range = lonmax - lonmin
        lat_range = latmax - latmin

        logger.debug(f"lon_range: {lon_range}, lat_range: {lat_range}")

        # handle polar stereographic projections differently
        if isinstance(
            projection,
            (ccrs.Stereographic, ccrs.NorthPolarStereo, ccrs.SouthPolarStereo),
        ):
            extent = _set_polar_extent(
                ax, lonmax, lonmin, latmax, latmin, projection, **kwargs
            )

        else:
            extent = [
                lonmin - lon_range * map_extent_multiplicator_lon,
                lonmax + lon_range * map_extent_multiplicator_lon,
                latmin - lat_range * map_extent_multiplicator_lat,
                latmax + lat_range * map_extent_multiplicator_lat,
            ]

        logger.debug(
            f"Final extent: {extent}"
        )  # BUG when latmin=40, and get Datarray with lon and lat range

        ax.set_extent(extent, crs=ccrs.PlateCarree())
    elif kwargs.get("map_extent_llon") is False:
        pass
    else:
        ax.set_extent(
            [
                kwargs.get("map_extent_llon"),
                kwargs.get("map_extent_ulon"),
                kwargs.get("map_extent_llat"),
                kwargs.get("map_extent_ulat"),
            ],
            crs=ccrs.PlateCarree(),
        )


def plot_poi(ax, poi, plons, plats, **kwargs):
    """
    Plots points of interest (POI) on the given axes.
    """
    plats = poi.vars.lats.data
    platsmax, platsmin = np.max(plats), np.min(plats)
    plons = poi.vars.lons.data
    plonsmax, plonsmin = np.max(plons), np.min(plons)
    # COMMENT: Why is the track plotted twice? Once as a line and once as points? Is this intentional?
    tc = ax.plot(
        plons,
        plats,
        color="cornflowerblue",
        ls="-",
        lw=1,
        zorder=-1,
        transform=ccrs.PlateCarree(),
    )  # use transform=ccrs.PlateCarree() to plot in lon/lat coordinates
    tc = ax.plot(
        plons,
        plats,
        color="cornflowerblue",
        ls="None",
        marker="o",
        ms=5,
        markeredgecolor="k",
        zorder=-1,
        transform=ccrs.PlateCarree(),
    )
    lonmax, lonmin = np.max([lonmax, plonsmax]), np.min([lonmin, plonsmin])
    latmax, latmin = np.max([latmax, platsmax]), np.min([latmin, platsmin])

    return lonmax, lonmin, latmax, latmin


def plot_var_field(
    ax, plot_var, lons, lats, vmin, vmax, cmap, cflevels, clevels, norm, **kwargs
):
    if len(plot_var.shape) > 1:
        sc = ax.contourf(
            lons.squeeze(),
            lats.squeeze(),
            plot_var.squeeze(),
            cmap=cmap,
            levels=cflevels,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            transform=ccrs.PlateCarree(),
            transform_first=kwargs.get("transform_first", False),
        )
        c = ax.contour(
            lons.squeeze(),
            lats.squeeze(),
            plot_var.squeeze(),
            levels=clevels,
            colors="w",
            linewidths=0.3,
            transform=ccrs.PlateCarree(),
            transform_first=kwargs.get("transform_first", False),
        )
    else:
        sc = ax.scatter(
            lons,
            lats,
            s=15,
            c=plot_var,
            marker="o",  # edgecolor='face',
            edgecolors="k",
            linewidths=0.1,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )
        c = None
    return sc, c


"""
from abc import abstractmethod


class Quicklook:
    @property
    @abstractmethod
    def projection():
        pass

    @property
    @abstractmethod
    def vars():
        pass

    def quicklook(self, blah):
        proj = self.projection

        for var in self.vars:
            # do something
            pass

class Sat(Quicklook):
    vars = [ "hs", "foo" ]
    projection = pyproj.Proj('+proj=latlong')
"""
