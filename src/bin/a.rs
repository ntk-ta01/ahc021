const N: usize = 30;
const N2: usize = N * (N + 1) / 2;
const MAX_TURN: usize = 10000;

fn main() {
    let input = parse_input();
    let mut state = State::new(input.bs);
    let mut out = vec![];
    greedy(&mut state, &mut out);
    write_output(&out);
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

fn greedy(state: &mut State, out: &mut Output) {
    for i in 0..N - 1 {
        for j in 0..=i {
            if state.bs[i + 1][j] < state.bs[i][j] || state.bs[i + 1][j + 1] < state.bs[i][j] {
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
            if out.len() == MAX_TURN {
                return;
            }
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

fn compute_score(input: &Input, out: &Output) -> (i64, Vec<Vec<i32>>) {
    let mut used = vec![vec![]; N];
    #[allow(clippy::needless_range_loop)]
    for i in 0..N {
        used[i] = vec![false; i + 1];
    }
    let mut bs = input.bs.clone();
    for (t, &(p, q)) in out.iter().enumerate() {
        if !is_adj(p, q) {
            panic!(
                "({}, {}) and ({}, {}) are not adjacent (turn {})",
                p.0, p.1, q.0, q.1, t
            );
        }
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
    let score = if num == 0 {
        ((10000 - out.len()) * 5 + 50000) as i64
    } else {
        50000 - num * 50
    };
    (score, bs)
}

fn is_adj((x1, y1): (usize, usize), (x2, y2): (usize, usize)) -> bool {
    if x1 == x2 {
        y1 == y2 + 1 || y1 + 1 == y2
    } else if x1 + 1 == x2 {
        y1 == y2 || y1 + 1 == y2
    } else if x1 == x2 + 1 {
        y1 == y2 || y1 == y2 + 1
    } else {
        false
    }
}
