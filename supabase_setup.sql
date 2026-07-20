-- Ejecutar en Supabase: Project -> SQL Editor -> New query

create table if not exists predicciones_log (
    id bigint generated always as identity primary key,
    fecha timestamp with time zone default timezone('utc'::text, now()) not null,
    tipo_prediccion text,          -- 'estimacion_monto' | 'clasificacion_infraccion' | 'estimacion_duracion'
    inputs_usuario jsonb,
    resultado_prediccion text
);

-- Índice para poder filtrar rápido por tipo de predicción en dashboards
create index if not exists idx_predicciones_tipo on predicciones_log (tipo_prediccion);
