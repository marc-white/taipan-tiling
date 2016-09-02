from taipan.simulate.simulate import test_redshift_success
import logging
import numpy as np

if __name__ == '__main__':
    # Generate test array
    testarr = np.zeros(30, dtype=[('target_id', 'int'),
                                  ('is_vpec_target', np.dtype(bool)),
                                  ('is_H0_target', np.dtype(bool)),
                                  ('is_lowz_target', np.dtype(bool))])
    for i in range(len(testarr)):
        testarr[i]['target_id'] = i
    visits = np.ones((30,), dtype=int)

    # Construct single-type targets
    # Recall the ordering is vpec, H0, lowz
    testarr[1]['is_H0_target'] = True
    testarr[2]['is_lowz_target'] = True
    testarr[3:5]['is_vpec_target'] = True
    for i in range(3,5):
        visits[i] = i - 3 + 1

    # Construct two-type targets
    testarr[5]['is_H0_target'] = True
    testarr[5]['is_lowz_target'] = True
    testarr[6:9]['is_H0_target'] = True
    testarr[6:9]['is_vpec_target'] = True
    for i in range(6,8):
        visits[i] = i - 6 + 1
    testarr[8:11]['is_lowz_target'] = True
    testarr[8:11]['is_vpec_target'] = True
    for i in range(8,11):
        visits[i] = i - 8 + 1

    # Construct all-type target
    testarr[11:14]['is_H0_target'] = True
    testarr[11:14]['is_vpec_target'] = True
    testarr[11:14]['is_lowz_target'] = True
    for i in range(11,14):
        visits[i] = i - 14 + 1

    # Trim the test arrays to what was used
    testarr = testarr[:14]
    visits = visits[:14]
    # print testarr
    # print visits

    # Run test_redshift_success across the system
    result = test_redshift_success(testarr, visits)
    print result