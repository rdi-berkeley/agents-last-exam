import unreal


def _parse_cmd() -> tuple[list[str], list[str], dict[str, str]]:
    return unreal.SystemLibrary.parse_command_line(unreal.SystemLibrary.get_command_line())


def _require_param(params: dict[str, str], key: str) -> str:
    value = params.get(key)
    if not value:
        raise RuntimeError(f"Missing '-{key}=...' argument")
    return value


def _optional_param(params: dict[str, str], key: str) -> str | None:
    value = params.get(key)
    if not value:
        return None
    return value


def _get_property_if_present(obj, key: str):
    try:
        return obj.get_editor_property(key)
    except Exception:
        return None


def _set_property_if_present(obj, key: str, value) -> bool:
    try:
        obj.set_editor_property(key, value)
        return True
    except Exception:
        return False


def _apply_conservative_render_settings(config) -> None:
    aa_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    _set_property_if_present(aa_setting, "bOverrideAntiAliasing", True)
    anti_aliasing_none = None
    for enum_name in ("AAM_NONE", "AAM_None", "None"):
        anti_aliasing_none = getattr(unreal.AntiAliasingMethod, enum_name, None)
        if anti_aliasing_none is not None:
            break
    if anti_aliasing_none is not None:
        _set_property_if_present(aa_setting, "AntiAliasingMethod", anti_aliasing_none)
    _set_property_if_present(aa_setting, "SpatialSampleCount", 1)
    _set_property_if_present(aa_setting, "TemporalSampleCount", 1)
    _set_property_if_present(aa_setting, "bRenderWarmUpFrames", False)
    _set_property_if_present(aa_setting, "bUseCameraCutForWarmUp", False)
    _set_property_if_present(aa_setting, "EngineWarmUpCount", 0)
    _set_property_if_present(aa_setting, "RenderWarmUpCount", 0)

    try:
        cvar_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    except Exception:
        cvar_setting = None
    if cvar_setting is not None:
        for name, value in (
            ("r.DefaultFeature.AntiAliasing", 0.0),
            ("r.PostProcessAAQuality", 0.0),
            ("r.TemporalAA.Upsampling", 0.0),
            ("r.Tonemapper.Sharpen", 0.0),
        ):
            try:
                cvar_setting.add_or_update_console_variable(name, value)
            except Exception:
                unreal.log_warning(
                    f"SceneRestorationRuntimeExecutor could not apply console variable override {name}={value}"
                )


@unreal.uclass()
class SceneRestorationRuntimeExecutor(unreal.MoviePipelinePythonHostExecutor):
    active_movie_pipeline = unreal.uproperty(unreal.MoviePipeline)

    def _post_init(self):
        self.active_movie_pipeline = None
        self._scene_restoration_sequence = None
        self._scene_restoration_output_dir = None
        self._scene_restoration_config_asset = None

    def _log_render_configuration(self, config) -> None:
        aa_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
        output_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
        cvar_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
        unreal.log(
            "SceneRestorationRuntimeExecutor render config: "
            f"sequence={self._scene_restoration_sequence} "
            f"output_dir={self._scene_restoration_output_dir} "
            f"config_asset={self._scene_restoration_config_asset or '<default>'}"
        )
        unreal.log(
            "SceneRestorationRuntimeExecutor anti-aliasing: "
            f"method={_get_property_if_present(aa_setting, 'AntiAliasingMethod')} "
            f"spatial={_get_property_if_present(aa_setting, 'SpatialSampleCount')} "
            f"temporal={_get_property_if_present(aa_setting, 'TemporalSampleCount')} "
            f"warmup_frames={_get_property_if_present(aa_setting, 'bRenderWarmUpFrames')} "
            f"engine_warmup={_get_property_if_present(aa_setting, 'EngineWarmUpCount')} "
            f"render_warmup={_get_property_if_present(aa_setting, 'RenderWarmUpCount')}"
        )
        unreal.log(
            "SceneRestorationRuntimeExecutor output settings: "
            f"dir={_get_property_if_present(output_setting, 'output_directory')} "
            f"format={_get_property_if_present(output_setting, 'file_name_format')}"
        )
        unreal.log(
            "SceneRestorationRuntimeExecutor console-variable setting present: "
            f"{cvar_setting is not None}"
        )

    @unreal.ufunction(override=True)
    def execute_delayed(self, in_pipeline_queue):
        try:
            (_, _, cmd_parameters) = _parse_cmd()
            level_sequence_path = _require_param(cmd_parameters, "LevelSequence")
            config_asset_path = _optional_param(cmd_parameters, "MoviePipelineConfig")
            output_dir = _require_param(cmd_parameters, "SceneRestorationOutputDir")

            self._scene_restoration_sequence = level_sequence_path
            self._scene_restoration_output_dir = output_dir
            self._scene_restoration_config_asset = config_asset_path

            unreal.log(
                "SceneRestorationRuntimeExecutor starting render job: "
                f"sequence={level_sequence_path} "
                f"output_dir={output_dir} "
                f"config_asset={config_asset_path or '<default>'}"
            )
            self.pipeline_queue = unreal.new_object(unreal.MoviePipelineQueue, outer=self)
            new_job = self.pipeline_queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
            new_job.sequence = unreal.SoftObjectPath(level_sequence_path)

            config = new_job.get_configuration()
            if config_asset_path:
                preset_asset = unreal.load_asset(config_asset_path)
                if preset_asset is None:
                    unreal.log_warning(
                        f"SceneRestorationRuntimeExecutor could not load pipeline config asset: {config_asset_path}. "
                        "Falling back to default render settings."
                    )
                else:
                    config.copy_from(preset_asset)

            _apply_conservative_render_settings(config)
            output_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
            output_path = unreal.DirectoryPath()
            output_path.path = output_dir
            output_setting.output_directory = output_path
            output_setting.file_name_format = "{sequence_name}"

            config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
            config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
            config.initialize_transient_settings()
            self._log_render_configuration(config)

            self.active_movie_pipeline = unreal.new_object(
                self.target_pipeline_class,
                outer=self.get_last_loaded_world(),
                base_type=unreal.MoviePipeline,
            )
            self.active_movie_pipeline.on_movie_pipeline_work_finished_delegate.add_function_unique(
                self, "on_movie_pipeline_finished"
            )
            self.active_movie_pipeline.initialize(new_job)
        except Exception as exc:
            unreal.log_error(
                "SceneRestorationRuntimeExecutor failed to initialize "
                f"sequence={self._scene_restoration_sequence} "
                f"output_dir={self._scene_restoration_output_dir} "
                f"config_asset={self._scene_restoration_config_asset or '<default>'} "
                f"error={exc}"
            )
            self.on_executor_errored()

    @unreal.ufunction(override=True)
    def on_begin_frame(self):
        super(SceneRestorationRuntimeExecutor, self).on_begin_frame()

    @unreal.ufunction(override=True)
    def is_rendering(self):
        return self.active_movie_pipeline is not None

    @unreal.ufunction(ret=None, params=[unreal.MoviePipelineOutputData])
    def on_movie_pipeline_finished(self, results):
        unreal.log(
            "Scene restoration render completed: "
            f"sequence={self._scene_restoration_sequence} success={results.success}"
        )
        self.active_movie_pipeline = None
        self.on_executor_finished_impl()
