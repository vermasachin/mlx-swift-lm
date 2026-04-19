module.exports = {
  apps: [
    {
      name: 'mlxserver-swift',
      script: '/Users/sachinverma/personal/mlx-swift-lm/run-server.sh',
      interpreter: 'bash',
      cwd: '/Users/sachinverma/personal/mlx-swift-lm',
      env: {
        PORT: '8091',
        KV_SCHEME: 'turbo4v2',
      },
      autorestart: true,
      max_restarts: 5,
      restart_delay: 3000,
      kill_timeout: 10000,
      out_file: '/Users/sachinverma/personal/mlx-swift-lm/logs/out.log',
      error_file: '/Users/sachinverma/personal/mlx-swift-lm/logs/err.log',
      merge_logs: true,
      time: true,
    },
  ],
};
