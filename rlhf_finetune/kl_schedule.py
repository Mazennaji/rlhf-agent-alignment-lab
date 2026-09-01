def linear_kl_schedule(start, end, total_steps):
    def schedule(step):
        progress = min(step / max(total_steps, 1), 1.0)
        return start + (end - start) * progress
    return schedule