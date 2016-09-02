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
    visits = np.zeros((30,), dtype=int)

    # Construct single-type targets
    # Recall the ordering is vpec, H0, lowz
    testarr[1]['is_H0_target'] = True
    testarr[2]['is_lowz_target'] = True
    testarr[3:6]['is_vpec_target'] = True
    for i in range(3,6):
        visits[i] = i - 3

    # Construct two-type targets
    testarr[6]['is_H0_target'] = True
    testarr[6]['is_lowz_target'] = True
    testarr[7:10]['is_H0_target'] = True
    testarr[7:10]['is_vpec_target'] = True
    for i in range(7,10):
        visits[i] = i - 7
    testarr[10:13]['is_lowz_target'] = True
    testarr[10:13]['is_vpec_target'] = True
    for i in range(10,13):
        visits[i] = i - 10

    # Construct all-type target
    testarr[14:17]['is_H0_target'] = True
    testarr[14:17]['is_vpec_target'] = True
    testarr[14:17]['is_lowz_target'] = True
    for i in range(14,17):
        visits[i] = i - 14

    # Trim the test arrays to what was used
    testarr = testarr[:17]
    visits = visits[:17]
    # print testarr
    # print visits

    # Run test_redshift_success across the system
    result = test_redshift_success(testarr, visits)
    print result