# Module to organize gridding data

# imports
import numpy as np
import tqdm
from wavy.grid_stats import apply_metric
from wavy.wconfig import load_or_default
import logging

validation_metric_abbreviations = load_or_default('validation_metrics.yaml')
variable_def = load_or_default('variable_def.yaml')

class gridder_class():

    @staticmethod
    def _normalize_varalias(varalias, logger):
        if isinstance(varalias, list):
            if len(varalias) > 1:
                logger.warning(
                    "Warning: gridder only expects one varalias.")
                logger.warning(
                    "First varalias selected as default: {}".format(
                        varalias[0]))
                logger.warning(
                    "If you want to select another variable, please "
                    + "specify with varalias argument.")
            return varalias[0]
        return varalias

    def _assign_var_metadata_from_def(self):
        self.units = variable_def[self.varalias].get('units')
        self.stdvarname = variable_def[self.varalias].get('standard_name')

    def _init_from_oco(self, oco):
        self._assign_var_metadata_from_def()
        self.olons = np.array(oco.vars['lons'].squeeze().values.ravel())
        self.olats = np.array(oco.vars['lats'].squeeze().values.ravel())
        self.ovals = np.array(oco.vars[self.varalias].squeeze().values.ravel())
        self.sdate = oco.vars['time'][0]
        self.edate = oco.vars['time'][-1]

    def _init_from_cco(self, cco):
        list_vars = list(cco.vars.variables)
        assert 'model_' + self.varalias in list_vars,\
                        "model_{}".format(self.varalias) +\
                                  " is missing in " +\
                                  "the dataset, if you would like to " +\
                                  "validate another variable, please " +\
                                  "specify with varalias."
        assert 'obs_' + self.varalias in list_vars,\
                      "obs_{}".format(self.varalias) +\
                                  " is missing in " +\
                                  "the dataset, if you would like to " +\
                                  "validate another variable, please " +\
                                  "specify with varalias."
        self.olons = np.array(cco.vars['obs_lons'])
        self.olats = np.array(cco.vars['obs_lats'])
        self.ovals = np.array(cco.vars['obs_'+self.varalias])
        self.mvals = np.array(cco.vars['model_'+self.varalias])
        self._assign_var_metadata_from_def()
        self.sdate = cco.vars['time'][0]
        self.edate = cco.vars['time'][-1]

    def _init_from_mco(self, mco):
        self.olons = np.array(mco.vars.lons.squeeze().values.flatten())
        self.olats = np.array(mco.vars.lats.squeeze().values.flatten())
        self.ovals = np.array(
                mco.vars[self.varalias].squeeze().values.flatten())
        self.stdvarname = mco.stdvarname
        self._assign_var_metadata_from_def()
        self.sdate = mco.vars['time'][0]
        self.edate = mco.vars['time'][-1]

    def _init_from_kwargs(self, kwargs):
        self.olons = kwargs.get('lons')
        self.olats = kwargs.get('lats')
        self.ovals = kwargs.get('values')
        self.stdvarname = kwargs.get('stdvarname', None)
        self.varalias = kwargs.get('varalias', None)
        self.units = kwargs.get('units', None)
        self.sdate = kwargs.get('sdate', None)
        self.edate = kwargs.get('edate', None)

    def __init__(
    self, oco=None, mco=None, cco=None, bb=None, grid='lonlat', res=(1, 1),
    **kwargs):
        """
        setup the gridder
        grid: lonlat or in m
        bb tupel: (lonmin, lonmax, latmin, latmax)
        res: resolution, tupel e.g. (.5,.5)
             in degree where tupel is (lon,lat) dimension
        """
        logger = logging.getLogger(__name__)
        log_level = str(kwargs.get('logging', 'WARNING').upper())
        logger.setLevel(getattr(logging, log_level, logging.WARNING))

        logger.info('# ----- ')
        logger.info(" ### Initializing gridder_class object ###")
        logger.info(" ")
        self.mvals = None
        if oco is not None:
            self.varalias = self._normalize_varalias(
                kwargs.get('varalias', oco.varalias), logger)
            self._init_from_oco(oco)
        elif cco is not None:
            self.varalias = self._normalize_varalias(
                kwargs.get('varalias', cco.varalias), logger)
            self._init_from_cco(cco)
        elif mco is not None:
            self.varalias = self._normalize_varalias(
                kwargs.get('varalias', mco.varalias), logger)
            self._init_from_mco(mco)
        else:
            self._init_from_kwargs(kwargs)

        self.bb = bb
        self.res = res
        self.grid = grid
        self.glons, self.glats = self.create_grid_coords()
        ovals, mvals, Midx = self.get_obs_grid_idx()
        self.ovals_clean = ovals
        self.mvals_clean = mvals
        self.Midx_clean = Midx
        logger.info(" ")
        logger.info(" ### gridder_class object initialized ###")
        logger.info('# ----- ')

    def create_grid_coords(self):
        """
        returns grid coordinates
        """
        lons = np.arange(self.bb[0], self.bb[1]+self.res[0]/2, self.res[0])
        lats = np.arange(self.bb[2], self.bb[3]+self.res[1]/2, self.res[1])
        return np.array(lons), np.array(lats)

    def get_obs_grid_idx(self):
        Midx = self.assign_obs_to_grid(
                self.glons, self.glats,
                self.olons, self.olats,
                self.res)
        ovals, mvals, Midx = self.clean_Midx(
                Midx, self.ovals, self.mvals, self.glons, self.glats)
        return ovals, mvals, Midx

    @staticmethod
    def assign_obs_to_grid(glons, glats, olons, olats, res):
        """
        assigns observation coordinates to grid indices
        """
        lonidx = np.floor((olons-np.min(glons))/res[0]).astype(int)
        latidx = np.floor((olats-np.min(glats))/res[1]).astype(int)
        Midx = np.array([lonidx, latidx], dtype=object)
        return Midx

    @staticmethod
    def clean_Midx(Midx, ovals, mvals, glons, glats):
        """
        cleans Midx and observations (ovals) for grid cells outside bb
        """
        # clean x-dim (tlons)
        glons_idx = np.where((Midx[0] >= 0) & (Midx[0] < len(glons)))[0]
        Midx = Midx[:, glons_idx]
        ovals = ovals[glons_idx]
        if mvals is not None:
            mvals = mvals[glons_idx]
        # clean y-dim (tlats)
        glats_idx = np.where((Midx[1] >= 0) & (Midx[1] < len(glats)))[0]
        Midx = Midx[:, glats_idx]
        ovals = ovals[glats_idx]
        if mvals is not None:
            mvals = mvals[glats_idx]
        return ovals, mvals, Midx

    @staticmethod
    def region_filter():
        # filter grid cells for region of interest
        # region could be rectangular, polygon, ...
        # -> return grid
        return

    @staticmethod
    def get_grid_idx(Midx):
        return np.where((Midx[0] == Midx[0][0]) & (Midx[1] == Midx[1][0]))[0]

    @staticmethod
    def calc_mean(gidx, ovals):
        return np.mean(np.array(ovals)[gidx])

    @staticmethod
    def rm_used_idx_from_Midx(gidx, Midx):
        d1 = np.delete(Midx[0, :], gidx)
        d2 = np.delete(Midx[1, :], gidx)
        return np.array([d1, d2], dtype=object)

    @staticmethod
    def get_exteriors(glons, glats, res):
        xb = np.array([glons+res[0], glons+res[0],
                       glons, glons,
                       glons+res[0]])
        yb = np.array([glats,
                       glats+res[1], glats+res[1],
                       glats, glats])
        return xb, yb

    @staticmethod
    def _format_title_date(value):
        """Return a stable string representation for mixed datetime-like types."""
        date_value = value
        if isinstance(date_value, np.datetime64):
            return str(date_value)
        if isinstance(date_value, np.ndarray) and date_value.shape == ():
            scalar_value = date_value[()]
            if isinstance(scalar_value, np.datetime64):
                return str(scalar_value)
            date_value = scalar_value
        elif isinstance(date_value, np.generic):
            date_value = date_value.item()
        elif hasattr(date_value, 'data') and not isinstance(date_value, np.ndarray):
            date_value = date_value.data
            if isinstance(date_value, np.datetime64):
                return str(date_value)
            if isinstance(date_value, np.ndarray) and date_value.shape == ():
                scalar_value = date_value[()]
                if isinstance(scalar_value, np.datetime64):
                    return str(scalar_value)
                date_value = scalar_value
        return str(date_value)

    @staticmethod
    def _estimate_projected_aspect(lonmin, lonmax, latmin, latmax,
                                   projection, data_crs):
        """Estimate width/height aspect in projection coordinates."""
        try:
            lons = np.linspace(lonmin, lonmax, 64)
            lats = np.linspace(latmin, latmax, 64)
            edge_lons = np.concatenate([
                lons,
                lons,
                np.full_like(lats, lonmin),
                np.full_like(lats, lonmax)])
            edge_lats = np.concatenate([
                np.full_like(lons, latmin),
                np.full_like(lons, latmax),
                lats,
                lats])
            pts = projection.transform_points(data_crs, edge_lons, edge_lats)
            x = pts[:, 0]
            y = pts[:, 1]
            finite = np.isfinite(x) & np.isfinite(y)
            if np.any(finite):
                x = x[finite]
                y = y[finite]
                xspan = np.nanmax(x) - np.nanmin(x)
                yspan = np.nanmax(y) - np.nanmin(y)
                if xspan > 0 and yspan > 0:
                    return float(xspan / yspan)
        except Exception:
            pass

        lat_mid = 0.5 * (latmin + latmax)
        lon_span = abs(lonmax - lonmin) * max(np.cos(np.deg2rad(lat_mid)), 1e-3)
        lat_span = max(abs(latmax - latmin), 1e-6)
        return float(lon_span / lat_span)

    @staticmethod
    def _set_extent_projected(ax, lonmin, lonmax, latmin, latmax,
                              projection, data_crs):
        """Set map limits in projection coordinates to avoid lon-wrap issues."""
        try:
            lons = np.linspace(lonmin, lonmax, 181)
            lats = np.linspace(latmin, latmax, 181)
            edge_lons = np.concatenate([
                lons,
                lons,
                np.full_like(lats, lonmin),
                np.full_like(lats, lonmax)])
            edge_lats = np.concatenate([
                np.full_like(lons, latmin),
                np.full_like(lons, latmax),
                lats,
                lats])
            pts = projection.transform_points(data_crs, edge_lons, edge_lats)
            x = pts[:, 0]
            y = pts[:, 1]
            finite = np.isfinite(x) & np.isfinite(y)
            if np.count_nonzero(finite) < 4:
                raise ValueError('Insufficient finite transformed boundary points')

            x = x[finite]
            y = y[finite]
            xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
            ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
            xpad = max((xmax - xmin) * 0.01, 1.0)
            ypad = max((ymax - ymin) * 0.01, 1.0)
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)
            return True
        except Exception:
            return False

    @staticmethod
    def _debug_array_range(values):
        arr = np.asarray(values)
        if arr.size == 0:
            return {'size': 0, 'min': None, 'max': None}
        finite = np.isfinite(arr)
        if not np.any(finite):
            return {'size': int(arr.size), 'min': None, 'max': None}
        arrf = arr[finite]
        return {
            'size': int(arr.size),
            'min': float(np.nanmin(arrf)),
            'max': float(np.nanmax(arrf))}

    def grid_view(self, metric, mask_metric_llim, mask_metric, **kwargs):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cmocean
        import matplotlib.pyplot as plt
        import matplotlib.cm as mplcm
        import matplotlib as mpl
        from copy import deepcopy

        # shift coords for plotting
        raw_lon_grid = kwargs.get('lon_grid')
        raw_lat_grid = kwargs.get('lat_grid')
        lon_grid = raw_lon_grid + self.res[0]/2.
        lat_grid = raw_lat_grid + self.res[1]/2.
        debug_plot = bool(kwargs.get('debug_plot', False))
        logger = logging.getLogger(__name__)

        # backup values
        all_grid = deepcopy(kwargs.get('val_grid'))
        mask_grid = all_grid[mask_metric]
        val_grid = all_grid[metric]

        # apply mask
        mask_llim_idx = np.where(mask_grid < mask_metric_llim)
        val_grid[mask_llim_idx[0], mask_llim_idx[1]] = np.nan

        projection = kwargs.get('projection', ccrs.PlateCarree())
        data_crs = kwargs.get('data_crs', ccrs.PlateCarree())
        norm = kwargs.get('norm')

        metric_meta = validation_metric_abbreviations.get(metric, {})
        metric_name = metric_meta.get('name', metric)
        is_bias_metric = (
            'bias' in str(metric).lower()
            or 'bias' in str(metric_name).lower())

        # parse kwargs
        if kwargs.get('cmap') is None:
            if is_bias_metric:
                cmap = getattr(cmocean.cm, 'balance', None)
                if cmap is None:
                    cmap = mplcm.get_cmap('RdBu_r')
            else:
                cmap = cmocean.cm.amp
        else:
            cmap = kwargs.get('cmap')

        if norm is None and is_bias_metric:
            finite_vals = val_grid[np.isfinite(val_grid)]
            if finite_vals.size > 0:
                absmax = np.nanmax(np.abs(finite_vals))
                if absmax > 0:
                    norm = mpl.colors.TwoSlopeNorm(
                        vmin=-absmax, vcenter=0.0, vmax=absmax)
                else:
                    norm = mpl.colors.CenteredNorm(vcenter=0.0)
            else:
                norm = mpl.colors.CenteredNorm(vcenter=0.0)

        # max/min for colorbar
        vmax = kwargs.get('vmax')
        vmin = kwargs.get('vmin')
        if norm is not None:
            # Matplotlib does not support passing vmin/vmax together with norm.
            vmin = None
            vmax = None

        # Determine extent from explicit kwargs or from unshifted grid/bbox.
        # Using shifted cell centers can push lonmax > 180 and trigger
        # dateline-wrap collapse in near-global stereographic plots.
        if kwargs.get('lonmin') is not None:
            lonmin = kwargs.get('lonmin')
        elif self.bb is not None:
            lonmin = self.bb[0]
        else:
            lonmin = np.min(raw_lon_grid)

        if kwargs.get('lonmax') is not None:
            lonmax = kwargs.get('lonmax')
        elif self.bb is not None:
            lonmax = self.bb[1]
        else:
            lonmax = np.max(raw_lon_grid)

        if kwargs.get('latmin') is not None:
            latmin = kwargs.get('latmin')
        elif self.bb is not None:
            latmin = self.bb[2]
        else:
            latmin = np.min(raw_lat_grid)

        if kwargs.get('latmax') is not None:
            latmax = kwargs.get('latmax')
        elif self.bb is not None:
            latmax = self.bb[3]
        else:
            latmax = np.max(raw_lat_grid)

        # land
        land = cfeature.GSHHSFeature(
                    scale=kwargs.get('land_mask_resolution', 'i'),
                    levels=[1],
                    facecolor=cfeature.COLORS['land'])

        map_rect = kwargs.get('map_rect', [0.08, 0.12, 0.72, 0.78])
        cbar_pad = kwargs.get('cbar_pad', 0.02)
        cbar_width = kwargs.get('cbar_width', 0.03)
        projected_aspect = self._estimate_projected_aspect(
            lonmin, lonmax, latmin, latmax, projection, data_crs)
        aspect_limited = np.clip(projected_aspect, 0.45, 2.8)
        base_height = kwargs.get('fig_height', 6.0)
        auto_fig_width = (
            base_height
            * aspect_limited
            * (map_rect[2] / max(map_rect[3], 1e-6)))
        fig_width = kwargs.get('fig_width', float(np.clip(auto_fig_width, 7.0, 14.0)))
        figsize = kwargs.get('figsize', (fig_width, base_height))

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(map_rect, projection=projection)
        # add land
        ax.add_feature(land, edgecolor='black', linewidth=1)

        use_projected_extent = kwargs.get('use_projected_extent', True)
        if use_projected_extent:
            extent_set = self._set_extent_projected(
                ax, lonmin, lonmax, latmin, latmax, projection, data_crs)
            if not extent_set:
                ax.set_extent([lonmin, lonmax, latmin, latmax], crs=data_crs)
        else:
            ax.set_extent([lonmin, lonmax, latmin, latmax], crs=data_crs)
        ax.set_aspect(kwargs.get('map_aspect', 'auto'))
        pc = ax.pcolormesh(
                lon_grid, lat_grid, val_grid,
                transform=data_crs, cmap=cmap,
                norm=norm, vmax=vmax, vmin=vmin)

        cax = fig.add_axes([
            map_rect[0] + map_rect[2] + cbar_pad,
            map_rect[1],
            cbar_width,
            map_rect[3]])

        metric_name = validation_metric_abbreviations[metric].get('name')
        metric_units =\
            validation_metric_abbreviations[metric].get('units', self.units)
        if metric_units is None:
            cbar = fig.colorbar(pc, cax=cax, label=metric_name)
        else:
            cbar = fig.colorbar(pc, cax=cax,
                                label=metric_name
                                + ' [' + metric_units + ']')

        # ax.coastlines()
        gl = ax.gridlines(draw_labels=True, crs=data_crs,
                          linewidth=1, color='grey', alpha=0.4,
                          linestyle='-')
        gl.top_labels = False
        gl.right_labels = False
        autotitle = ('Base variable: ' + self.varalias + '\n'
                     + 'from ' + self._format_title_date(self.sdate)
                     + ' to ' + self._format_title_date(self.edate))
        if kwargs.get('title') is None:
            ax.set_title(autotitle)
        else:
            ax.set_title(kwargs.get('title'))
        ax.title.set_size(11)

        if debug_plot:
            debug_payload = {
                'metric': metric,
                'bb': self.bb,
                'res': self.res,
                'projection': str(projection),
                'data_crs': str(data_crs),
                'use_projected_extent': use_projected_extent,
                'projected_extent_set': bool(
                    use_projected_extent and 'extent_set' in locals() and extent_set),
                'requested_extent_lonlat': {
                    'lonmin': float(lonmin),
                    'lonmax': float(lonmax),
                    'latmin': float(latmin),
                    'latmax': float(latmax)},
                'raw_lon_grid': self._debug_array_range(raw_lon_grid),
                'raw_lat_grid': self._debug_array_range(raw_lat_grid),
                'shifted_lon_grid': self._debug_array_range(lon_grid),
                'shifted_lat_grid': self._debug_array_range(lat_grid),
                'figure_size_inches': tuple(float(v) for v in fig.get_size_inches()),
                'map_rect': tuple(float(v) for v in map_rect),
                'ax_position': tuple(float(v) for v in ax.get_position().bounds),
                'ax_xlim': tuple(float(v) for v in ax.get_xlim()),
                'ax_ylim': tuple(float(v) for v in ax.get_ylim()),
                'cax_position': tuple(float(v) for v in cax.get_position().bounds)}

            self.last_grid_view_debug = debug_payload
            logger.warning('grid_view diagnostic payload: %s', debug_payload)

        # todo: add info on observation and model source for figure
        if kwargs.get('show', True):
            plt.show()
        return fig

    def quicklook(self, metric='mor',
                  mask_metric_llim=10,
                  mask_metric='nov',
                  **kwargs):

        if metric == 'all':
            figs = {}
            for key in kwargs['val_grid'].keys():
                figs[key] = self.grid_view(
                    key, mask_metric_llim, mask_metric, **kwargs)
            return figs
        else:
            return self.grid_view(metric, mask_metric_llim, mask_metric, **kwargs)
