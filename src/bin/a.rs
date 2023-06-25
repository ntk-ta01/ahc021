use rand::prelude::*;
const N: usize = 30;
// const N2: usize = N * (N + 1) / 2;
const MAX_TURN: usize = 10000;
const TIMELIMIT: f64 = 1.9;
fn main() {
    let mut timer = Timer::new();
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(0);
    let input = parse_input();
    let mut out = vec![];
    let mut poses: Vec<_> = vec![];
    for i in 0..N - 1 {
        for j in 0..=i {
            poses.push((i, j));
        }
    }
    let mut state = State::new(input.bs.clone());
    greedy(&mut state, &mut out, &poses);
    annealing(&input, &mut out, &mut poses, &mut timer, &mut rng);
    write_output(&out);
}

fn annealing(
    input: &Input,
    output: &mut Output,
    poses: &mut Vec<(usize, usize)>,
    timer: &mut Timer,
    rng: &mut rand_pcg::Pcg64Mcg,
) {
    const T0: f64 = 10000.0;
    const T1: f64 = 0.01;
    let mut temp;
    let mut prob;

    let mut now_score = compute_score(input, output);

    let mut best_score = now_score;
    let mut best_output = output.clone();

    loop {
        let passed = timer.get_time() / TIMELIMIT;
        if passed >= 1.0 {
            break;
        }
        temp = T0.powf(1.0 - passed) * T1.powf(passed);

        let mut new_out = vec![];
        let mut new_state = State::new(input.bs.clone());
        let mut new_poses = poses.clone();
        // 近傍解生成。
        // posesの近傍を見てgreedyする
        let i = rng.gen_range(0, new_poses.len());
        let j = rng.gen_range(0, new_poses.len());
        new_poses.swap(i, j);
        greedy(&mut new_state, &mut new_out, &new_poses);

        let new_score = compute_score(input, &new_out);
        prob = f64::exp((new_score - now_score) as f64 / temp);
        if now_score <= new_score || rng.gen_bool(prob) {
            now_score = new_score;
            *poses = new_poses;
            *output = new_out;
        }

        if best_score < now_score {
            best_score = now_score;
            best_output = output.clone();
        }
    }
    *output = best_output;
    // eprintln!("{}", best_score);
}

#[derive(Debug, Clone)]
struct State {
    bs: Vec<Vec<i32>>,
}

impl State {
    fn new(bs: Vec<Vec<i32>>) -> Self {
        State { bs }
    }
}

fn greedy(state: &mut State, out: &mut Output, poses: &[(usize, usize)]) {
    while out.len() < MAX_TURN {
        let mut no_changed = true;
        for &(i, j) in poses.iter() {
            if state.bs[i + 1][j] < state.bs[i][j] || state.bs[i + 1][j + 1] < state.bs[i][j] {
                no_changed = false;
                if state.bs[i + 1][j] < state.bs[i + 1][j + 1] {
                    let tmp = state.bs[i + 1][j];
                    state.bs[i + 1][j] = state.bs[i][j];
                    state.bs[i][j] = tmp;
                    out.push(((i, j), (i + 1, j)));
                } else {
                    let tmp = state.bs[i + 1][j + 1];
                    state.bs[i + 1][j + 1] = state.bs[i][j];
                    state.bs[i][j] = tmp;
                    out.push(((i, j), (i + 1, j + 1)));
                }
            }
        }
        if no_changed {
            break;
        }
    }
}

type Output = Vec<((usize, usize), (usize, usize))>;

fn write_output(output: &Output) {
    println!("{}", output.len());
    for &((x1, y1), (x2, y2)) in output.iter() {
        println!("{} {} {} {}", x1, y1, x2, y2);
    }
}

#[derive(Clone, Debug)]
pub struct Input {
    pub bs: Vec<Vec<i32>>,
}

fn parse_input() -> Input {
    use proconio::input;
    let mut bs = vec![];
    for i in 0..N {
        input! {
            b: [i32; i + 1]
        }
        bs.push(b);
    }
    Input { bs }
}

fn compute_score(input: &Input, out: &Output) -> i64 {
    let mut bs = input.bs.clone();
    for (_t, &(p, q)) in out.iter().enumerate() {
        let bp = bs[p.0][p.1];
        let bq = bs[q.0][q.1];
        bs[p.0][p.1] = bq;
        bs[q.0][q.1] = bp;
    }
    let mut num = 0;
    for x in 0..N - 1 {
        for y in 0..=x {
            if bs[x][y] > bs[x + 1][y] {
                num += 1;
            }
            if bs[x][y] > bs[x + 1][y + 1] {
                num += 1;
            }
        }
    }
    if num == 0 {
        ((10000 - out.len()) * 5 + 50000) as i64
    } else {
        50000 - num * 50
    }
}

fn get_time() -> f64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9
}

struct Timer {
    start_time: f64,
}

impl Timer {
    fn new() -> Timer {
        Timer {
            start_time: get_time(),
        }
    }

    fn get_time(&self) -> f64 {
        get_time() - self.start_time
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.start_time = 0.0;
    }
}
